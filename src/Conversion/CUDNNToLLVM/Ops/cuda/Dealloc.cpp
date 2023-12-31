//===----------------------------------------------------------------------===//
// CUDNN to LLVM: CUDNNDeallocOpLowering
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/CUDNNToLLVM/CUDNNRuntimeAPI.hpp"
#include "src/Conversion/CUDNNToLLVM/CUDNNToLLVMCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class CUDNNDeallocOpLowering : public ConvertToLLVMPattern {
public:
  CUDNNDeallocOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::CUDNNDeallocOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = CUDNNRuntimeAPIRegistry(module, rewriter, inType);

    auto operand = op->getOperand(0);

    auto int8Ty = IntegerType::get(context, 8);
    auto int8PtrTy = LLVM::LLVMPointerType::get(int8Ty);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);

    auto memRefType = operand.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmelemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto allocPtrType = LLVM::LLVMPointerType::get(llvmelemType);

    Value convertedOperand = castToLLVMStruct(context, typeConverter, rewriter, loc, operand);
    auto extractPtr = rewriter.create<LLVM::ExtractValueOp>(loc, allocPtrType, convertedOperand,
        llvm::ArrayRef<int64_t>{0});
    auto bitcast = rewriter.create<LLVM::BitcastOp>(loc, int8PtrTy, extractPtr);
    auto callFree = CUDNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, CUDNNRuntimeAPI::API::CUDNN_DEALLOC, {bitcast});

    rewriter.replaceOp(op, callFree);

    return success();
  }
};

void mlir::populateCUDNNDeallocToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<CUDNNDeallocOpLowering>(ctx, typeConverter);
}
