#ifndef __CONVERT_ONNX_TO_CUDNN_H__
#define __CONVERT_ONNX_TO_CUDNN_H__

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Pass/CDPasses.hpp"
#include "src/Support/KrnlSupport.hpp"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;
using namespace onnx_mlir;

/*
void dbgPrint(){}

template<typename First, typename ...Rest>
void dbgPrint(First && first, Rest && ...rest)
{
  if (DEBUG) {
      std::cout << std::forward<First>(first);
      dbgPrint(std::forward<Rest>(rest)...);
   }
}
*/

/// Get the corresponding MemRefType of a given TensorType/MemRefType.
MemRefType convertToMemRefType(Type type);

// Compute broadcasting Dim for B at (A + B) instruction.
// not compute the whole dimension, just make rankB match with rankA.
void broadcastOnlyDimension (SmallVector<int64_t>* broadcastedDim,
    MemRefType inputAMemRef, MemRefType inputBMemRef);

// Determine if current function returns the result value of the
// current op being lowered. If it does then the result value must
// be copied into host memory.
bool checkInsertMemcpy(Operation *currentOp, int resultIndex = 0);

// Allocate a MemRef and copy the result of the current op from device
// memory to the MemRef.
Value insertMemcpyToHost(Operation* op, Value result, Location loc, PatternRewriter &rewriter,
    Value operand = nullptr, int64_t alignment = -1);

// Insert cudnn.dealloc op for the current op. All ops including
// the returning op requires a dealloc, because this is about device
// memory.
Value insertDealloc(Value alloc, Location loc, PatternRewriter &rewriter);


// Emit constant operation.
Value emitConstantOp(OpBuilder &rewriter, Location loc, Type type, double value);

//===----------------------------------------------------------------------===//
// Conversion from Tensor type to the Standard dialect MemRef type.
//===----------------------------------------------------------------------===//

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  TensorTypeConverter() { addConversion(convertType); }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    if (auto type = convertToMemRefType(t)) {
      results.push_back(type);
      return success();
    }

    results.push_back(t);
    return success();
  }

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(FunctionType funcType) {
    return llvm::all_of(
        llvm::concat<const Type>(funcType.getInputs(), funcType.getResults()),
        [this](Type type) { return isLegal(type); });
  }

  /// Return true if the operands/results of call have a legal type.
  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

//===----------------------------------------------------------------------===//
// Populate lowering patterns.
//===----------------------------------------------------------------------===//

// ------------------ COMMON -------------------//
void populateLoweringONNXConstantOfShapeOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXConstantOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);

// ------------------- CUDNN -------------------//
void populateLoweringONNXConvOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXReluOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXSigmoidOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXTanhOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXAddOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXSubOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXMulOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXSqrtOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXReduceMeanOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXReduceMeanV13OpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXMaxPoolSingleOutOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXAveragePoolOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXSoftmaxOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);

// ------------------- cuBLAS -------------------//
void populateLoweringONNXGemmOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXMatMulOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);

// -------------------- CUDA --------------------//
void populateLoweringONNXCastOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXClipOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXGatherOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXExpandOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXConcatOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXReshapeOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXNonZeroOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXFlattenOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXUnsqueezeOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXUnsqueezeV11OpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXSqueezeOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXSqueezeV11OpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXTransposeOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXDivOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXPowOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXErfOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXPReluOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXPadOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);
void populateLoweringONNXLeakyReluOpToCUDNNPattern(
        RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context);

#endif // __CONVERT_ONNX_TO_CUDNN_H__


