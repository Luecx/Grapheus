#pragma once

#include "distribute/distribute.h"
#include "distribute/distribute_bp.h"
#include "select/select.h"
#include "select/select_bp.h"
#include "select_single/select_single.h"
#include "select_single/select_single_bp.h"

#include "relu/relu_bp.h"
#include "relu/relu.h"
#include "crelu/crelu_bp.h"
#include "crelu/crelu.h"
#include "lrelu/lrelu_bp.h"
#include "lrelu/lrelu.h"
#include "sigmoid/sigmoid_bp.h"
#include "sigmoid/sigmoid.h"

#include "affine/affine_bp.h"
#include "affine/affine.h"
#include "affine_batched/affine_batched.h"
#include "affine_batched/affine_batched_bp.h"
#include "affine_sparse/affine_sparse.h"
#include "affine_sparse/affine_sparse_bp.h"
#include "affine_sparse_shared/affine_sparse_shared.h"
#include "affine_sparse_shared/affine_sparse_shared_bp.h"

#include "ax_p_by/ax_p_by.h"
#include "ax_p_by/ax_p_by_bp.h"

#include "mean_power_error/mpe.h"

#include "adam/adam.h"