//===------------ Converting ONNX to CUDNN -----------===//

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"

using namespace mlir;
using namespace onnx_mlir;

//========== Start of ConvertONNXToCUDNNPass ==========//
namespace {
struct ConvertONNXToCUDNNPass
    : public PassWrapper<ConvertONNXToCUDNNPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-onnx-to-cudnn"; }

  StringRef getDescription() const override {
    return "Lower the ONNX dialects to CUDNN.";
  }
  void runOnOperation() final;
};
} // end of namespace for ConvertONNXToCUDNNPass

void ConvertONNXToCUDNNPass::runOnOperation() {

  ModuleOp module = getOperation();
  ConversionTarget target(getContext());

  target.addLegalDialect<CUDNNDialect, KrnlDialect, arith::ArithDialect, memref::MemRefDialect>();

  RewritePatternSet patterns(&getContext());

  // Convert TensorType to MemRef
  onnx_mlir::KrnlTypeConverter krnlTypeConverter;
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // func::FuncOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isLegal(op);
  });

  // Operations that are legal only if types are not tensors.
  target.addDynamicallyLegalOp<func::ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
      [](Type type) { return type.isa<TensorType>();  });
  });

  // Type conversion for function signatures.
  // Call MLIR func::FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, krnlTypeConverter);
  populateCallOpTypeConversionPattern(patterns, krnlTypeConverter);
  populateReturnOpTypeConversionPattern(patterns, krnlTypeConverter);

  // ----------- Adding Patterns for Lowering Pass ----------- //

  // ===------------------ Constants ---------------------===//
  populateLoweringONNXConstantOfShapeOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXConstantOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());

  // ===----------------- CUDNN -------------------=== //
  populateLoweringONNXConvOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReluOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSigmoidOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXTanhOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXAddOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSubOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXMulOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSqrtOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReduceMeanOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReduceMeanV13OpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXMaxPoolSingleOutOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXAveragePoolOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSoftmaxOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());


  // ===---------------- cuBLAS -------------------=== //
  populateLoweringONNXGemmOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXMatMulOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());

  // ===----------------- CUDA --------------------=== //
  populateLoweringONNXCastOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXClipOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXGatherOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXExpandOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXConcatOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReshapeOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXNonZeroOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXFlattenOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXUnsqueezeOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXUnsqueezeV11OpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSqueezeOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSqueezeV11OpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXTransposeOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXDivOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXPowOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXErfOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXPReluOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXPadOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXLeakyReluOpToCUDNNPattern(patterns, krnlTypeConverter, &getContext());

  // --------------------------------------------------------- //

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> core_dnn::createConvertONNXToCUDNNPass() {
  return std::make_unique<ConvertONNXToCUDNNPass>();
}
//=========== End of ConvertONNXToCUDNNPass ===========//

