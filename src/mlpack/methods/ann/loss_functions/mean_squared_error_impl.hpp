/**
 * @file methods/ann/loss_functions/mean_squared_error_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the mean squared error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_squared_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanSquaredError<InputDataType, OutputDataType>
  ::MeanSquaredError(const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
MeanSquaredError<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  typename PredictionType::elem_type lossSum =
      arma::accu(arma::square(prediction - target));

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void MeanSquaredError<InputDataType, OutputDataType>::Backward(
    const PredictionType& prediction,
    const TargetType& target,
    LossType& loss)
{
  loss = 2 * (prediction - target);

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanSquaredError<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace ann
} // namespace mlpack

#endif
