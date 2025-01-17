/**
 * @file methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * Definition of the cross-entropy with logits performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The SigmoidCrossEntropyError performance function measures the network's
 * performance according to the cross-entropy function between the input and
 * target distributions. This function calculates the cross entropy
 * given the real values instead of providing the sigmoid activations.
 * The function uses this equivalent formulation:
 * \f$max(x, 0) - x * z + \log(1 +  e^{-|x|})\f$
 * where x = input and z = target.
 *
 * For more information, see the following paper.
 *
 * @code
 * @article{Janocha2017
 *   title   = {On Loss Functions for Deep Neural Networks in Classification},
 *   author  = {Katarzyna Janocha, Wojciech Marian Czarnecki},
 *   url     = {http://arxiv.org/abs/1702.05659},
 *   journal = {CoRR},
 *   eprint  = {arXiv:1702.05659},
 *   year    = {2017}
 * }
 * @endcode
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
class SigmoidCrossEntropyError
{
 public:
  /**
   * Create the SigmoidCrossEntropyError object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */                  
  SigmoidCrossEntropyError(const bool reduction = true);

  /**
   * Computes the Sigmoid CrossEntropy Error functions.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   */
  template<typename PredictionType, typename TargetType>
  inline typename PredictionType::elem_type Forward(
      const PredictionType& prediction,
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
  inline void Backward(const PredictionType& prediction,
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
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class SigmoidCrossEntropy

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "sigmoid_cross_entropy_error_impl.hpp"

#endif
