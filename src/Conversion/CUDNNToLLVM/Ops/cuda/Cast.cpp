//===----------------------------------------------------------------------===//
// CUDNN to LLVM: CUDNNCastOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/CUDNNToLLVM/CUDNNRuntimeAPI.hpp"
#include "src/Conversion/CUDNNToLLVM/CUDNNToLLVMCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class CUDNNCastOpLowering : public ConvertToLLVMPattern {
public:
  CUDNNCastOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::CUDNNCastOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    CUDNNCastOp castOp = dyn_cast<CUDNNCastOp>(op);

    mlir::Type inType = op->getOperand(0).getType();

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    const auto &apiRegistry = CUDNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);

    auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(0));

    auto input = castOp.getX();
    auto output = castOp.getY();
    auto outputDimAttr = castOp.getDimY();
    auto toAttr = castOp.getTo();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    auto outputllvmPtrType = LLVM::LLVMPointerType::get(
        typeConverter->convertType(
          typeConverter->convertType(
            output.getType().cast<mlir::MemRefType>().getElementType()
            )
          ).cast<mlir::Type>()
        );


    int inputRank = inputMemRefShape.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, outputllvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create 'to' constant for cast operation
    // XXX : TODO : XXX -> Now only support int64 to float casting
    rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(toAttr));

    // Create integer array from shape attribute
    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputRank));
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < inputRank; i++) { // inputRank == outputRank for CastOp
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputDimAttr[i].cast<IntegerAttr>());
      auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    auto callCast = CUDNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, CUDNNRuntimeAPI::API::CUDA_CAST,
        {extractInput, extractOutput, outputDim, rankConst});

    Value callCastOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callCast);

    rewriter.replaceOp(op, callCastOutput);

    return success();
  }
};

void mlir::populateCUDNNCastToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<CUDNNCastOpLowering>(ctx, typeConverter);
}
