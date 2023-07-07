//===----------------------------------------------------------------------===//
// CUDNN to LLVM: CUDNNNonZeroOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/CUDNNToLLVM/CUDNNRuntimeAPI.hpp"
#include "src/Conversion/CUDNNToLLVM/CUDNNToLLVMCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class CUDNNNonZeroOpLowering : public ConvertToLLVMPattern {
public:
  CUDNNNonZeroOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::CUDNNNonZeroOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    CUDNNNonZeroOp nonzeroOp = dyn_cast<CUDNNNonZeroOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::CUDNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);

    auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(0));

    auto input = nonzeroOp.getX();
    auto output = nonzeroOp.getY();
    auto inputDimAttr = nonzeroOp.getDimX();
    auto outputDimAttr = nonzeroOp.getDimY();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    auto inputMemRefRank = inputMemRefShape.size();

   // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputMemRefRank));
    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (long unsigned i = 0; i < inputMemRefRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          inputDimAttr[i].cast<IntegerAttr>());
      auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);

      auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputDimAttr[i].cast<IntegerAttr>());
      auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    Value callNonZero;
    if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
      callNonZero = core_dnn::CUDNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, CUDNNRuntimeAPI::API::CUDA_NONZERO_I64,
          {extractInput, inputDim, extractOutput, outputDim, rankConst});
    } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
      callNonZero = core_dnn::CUDNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, CUDNNRuntimeAPI::API::CUDA_NONZERO_F32,
          {extractInput, inputDim, extractOutput, outputDim, rankConst});
    }

    Value callNonZeroOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callNonZero);

    rewriter.replaceOp(op, callNonZeroOutput);

    return success();
  }
};

void mlir::populateCUDNNNonZeroToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<CUDNNNonZeroOpLowering>(ctx, typeConverter);
}
