/**
 * @file methods/ann/loss_functions/margin_ranking_loss.hpp
 * @author Andrei Mihalea
 *
 * Definition of the Margin Ranking Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ANN_LOSS_FUNCTION_MARGIN_RANKING_LOSS_HPP
#define MLPACK_ANN_LOSS_FUNCTION_MARGIN_RANKING_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Margin ranking loss measures the loss given inputs and a label vector with
 * values of 1 or -1. If the label is 1 then the first input should be ranked
 * higher than the second input at a distance larger than a margin, and vice-
 * versa if the label is -1.
 * 
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MarginRankingLoss
{
 public:
  /**
   * Create the MarginRankingLoss object with Hyperparameter margin.
   * @param margin defines a minimum distance between correctly ranked samples.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  MarginRankingLoss(const double margin = 1.0, const bool reduction = true);

  /**
   * Computes the Margin Ranking Loss function.
   * 
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The label vector which contains values of -1 or 1.
   */
  template<typename PredictionType, typename TargetType>
  typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                             const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The label vector which contains -1 or 1 values.
   * @param loss The calculated error.
   */
  template <
    typename PredictionType,
    typename TargetType,
    typename LossType
  >
  void Backward(const PredictionType& prediction,
                const TargetType& target,
                LossType& loss);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the margin parameter.
  double Margin() const { return margin; }
  //! Modify the margin parameter.
  double& Margin() { return margin; }

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The margin value used in calculating Margin Ranking Loss.
  double margin;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class MarginRankingLoss

} // namespace ann
} // namespace mlpack

// include implementation.
#include "margin_ranking_loss_impl.hpp"

#endif
