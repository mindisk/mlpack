/**
 * @file methods/ann/loss_functions/soft_margin_loss.hpp
 * @author Anjishnu Mukherjee
 *
 * Definition of the Soft Margin Loss function.
 *
 * It is a criterion that optimizes a two-class classification logistic loss,
 * between input x and target y, both having the same shape, with the target
 * containing only the values 1 or -1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_HPP
#define MLPACK_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class SoftMarginLoss
{
 public:
  /**
   * Create the SoftMarginLoss object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If
   *                  true, 'sum' reduction is used and the output will be
   *                  summed. It is set to true by default.
   */
  SoftMarginLoss(const bool reduction = true);

  /**
   * Computes the Soft Margin Loss function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector with same shape as input.
   */
  template<typename PredictionType, typename TargetType>
  typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                             const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  template<typename PredictionType, typename TargetType, typename LossType>
  void Backward(const PredictionType& prediction,
                const TargetType& target,
                LossType& loss);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class SoftMarginLoss

} // namespace ann
} // namespace mlpack

// include implementation.
#include "soft_margin_loss_impl.hpp"

#endif
