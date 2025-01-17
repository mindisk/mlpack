/**
 * @file methods/ann/loss_functions/log_cosh_loss.hpp
 * @author Kartik Dutt
 *
 * Definition of the Log-Hyperbolic-Cosine loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_LOG_COSH_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_LOG_COSH_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Log-Hyperbolic-Cosine loss function is often used to improve
 * variational auto encoder. This function is the log of hyperbolic
 * cosine of difference between true values and predicted values.
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
class LogCoshLoss
{
 public:
  /**
   * Create the Log-Hyperbolic-Cosine object with the specified
   * parameters.
   *
   * @param a A double type value for smoothening loss function. It must be a
   *          positive real number. Sharpness of loss function is directly
   *          proportional to a. It can also act as a scaling factor, hence
   *          making the loss function more sensitive to small losses around
   *          the origin. Default value = 1.0.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  LogCoshLoss(const double a = 1.0, const bool reduction = true);

  /**
   * Computes the Log-Hyperbolic-Cosine loss function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target Target data to compare with.
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

  //! Get the value of hyperparameter a.
  double A() const { return a; }
  //! Modify the value of hyperparameter a.
  double& A() { return a; }

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the loss function.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Hyperparameter a for smoothening function curve.
  double a;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class LogCoshLoss

} // namespace ann
} // namespace mlpack

// include implementation
#include "log_cosh_loss_impl.hpp"

#endif
