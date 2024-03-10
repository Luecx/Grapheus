#pragma once

#include "gradient_operation.h"

#include "distribute/distribute.h"
#include "distribute/distribute_bp.h"
#include "select/select.h"
#include "select/select_bp.h"
#include "select_single/select_single.h"
#include "select_single/select_single_bp.h"

#include "activation/activations.h"

#include "affine/affine_bp.h"
#include "affine/affine.h"
#include "affine/affine_bp.h"
#include "affine_batched/affine_batched.h"
#include "affine_batched/affine_batched_bp.h"
#include "affine_sparse/affine_sparse.h"
#include "affine_sparse/affine_sparse_bp.h"

#include "ax_p_by/ax_p_by.h"
#include "ax_p_by/ax_p_by_bp.h"
#include "elemwise_mul/elemwise_mul.h"
#include "elemwise_mul/elemwise_mul_bp.h"

#include "mean_power_error/mpe.h"
#include "select/select.h"
#include "select/select_bp.h"
#include "select_single/select_single.h"
#include "select_single/select_single_bp.h"

#include "adam/adam.h"