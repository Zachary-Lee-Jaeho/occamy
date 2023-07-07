//===----------------------------------------------------------------------===//
// CUDNN to LLVM: CUDNNMallocOpLowering
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/CUDNNToLLVM/CUDNNRuntimeAPI.hpp"
#include "src/Conversion/CUDNNToLLVM/CUDNNToLLVMCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class CUDNNMallocOpLowering : public ConvertToLLVMPattern {
public:
  CUDNNMallocOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::CUDNNMallocOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::CUDNNRuntimeAPIRegistry(module, rewriter, inType);

    auto operand = op->getOperand(0);

    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto opaquePtrPtrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);

    auto memRefType = op->getResult(0).getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto memRefShape = memRefType.getShape();
    auto llvmelemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto allocPtrType = LLVM::LLVMPointerType::get(llvmelemType);
    auto allocPtrPtrType = LLVM::LLVMPointerType::get(allocPtrType);

    // common constantOp
    auto zero64 = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(0));
    auto one32 = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));

    auto devPtr = rewriter.create<LLVM::AllocaOp>(loc, allocPtrPtrType, one32, 0);
    auto devPtrAddr = rewriter.create<LLVM::BitcastOp>(loc, opaquePtrPtrTy, devPtr);

    int64_t mallocSize =
      dyn_cast<arith::ConstantOp>(operand.getDefiningOp()).getValue().cast<IntegerAttr>().getInt();

    auto mallocSizeOp = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(mallocSize));

    auto callMalloc = CUDNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, CUDNNRuntimeAPI::API::CUDA_MALLOC,
        {devPtrAddr, mallocSizeOp});

    auto loadPtr = rewriter.create<LLVM::LoadOp>(loc, allocPtrType, devPtr);

    Value callMallocOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), loadPtr);

    rewriter.replaceOp(op, callMallocOutput);

    return success();
  }
};

void mlir::populateCUDNNMallocToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<CUDNNMallocOpLowering>(ctx, typeConverter);
}
