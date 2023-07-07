
//===------------ CUDNN Ops Header --------------===//
//===---------  XXX corelab Jaeho XXX -----------===//
//===--------------------------------------------===//

#ifndef __CUDNN_OPS_H__
#define __CUDNN_OPS_H__

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/CUDNN/CUDNNDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Dialect/CUDNN/CUDNNOps.hpp.inc"

#endif
