/**
 * This layer implements the 'reparameterization trick' described in
 * https://blog.keras.io/building-autoencoders-in-keras.html.
 * 
 * Source: https://github.com/tensorflow/tfjs-examples/blob/aad9b8151458ef6a8d00bcbaa6f6cb12584401af/fashion-mnist-vae/model.js#L75
 *
 * The implementation is in the call method.
 * Instead of sampling from Q(z|X):
 *    sample epsilon = N(0,I)
 *    z = z_mean + sqrt(var) * epsilon
 */
class ZLayer extends tf.layers.Layer {
    constructor(config) {
      super(config);
    }
  
    computeOutputShape(inputShape) {
      tf.util.assert(inputShape.length === 2 && Array.isArray(inputShape[0]),
          () => `Expected exactly 2 input shapes. But got: ${inputShape}`);
      return inputShape[0];
    }
  
    /**
     * The actual computation performed by an instance of ZLayer.
     *
     * @param {Tensor[]} inputs this layer takes two input tensors, z_mean and
     *     z_log_var
     * @return A tensor of the same shape as z_mean and z_log_var, equal to
     *     z_mean + sqrt(exp(z_log_var)) * epsilon, where epsilon is a random
     *     vector that follows the unit normal distribution (N(0, I)).
     */
    call(inputs, kwargs) {
      const [zMean, zLogVar] = inputs;
      const batch = zMean.shape[0];
      const dim = zMean.shape[1];
  
      const mean = 0;
      const std = 1.0;
      // sample epsilon = N(0, I)
      const epsilon = tf.randomNormal([batch, dim], mean, std);
  
      // z = z_mean + sqrt(var) * epsilon
      return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon));
    }
  
    static get className() {
      return 'ZLayer';
    }
  }

  tf.serialization.registerClass(ZLayer);