/**
 * Soft Actor Critic Agent https://arxiv.org/abs/1812.05905
 * without value network.
 */
class AgentSac {
    constructor({
        nActions = 9, 
        shape = [128, 256, 3], 
        batchSize = 1, 
        nFrames = 1,
        noise = 1e-6,
        gamma = 0.99,
        tau = 5e-3,
        rewardScale = 2
    }) {
        this.nActions = nActions
        this.shape = shape 
        this.batchSize = batchSize
        this.nFrames = nFrames
        this.noise = noise
        this.gamma = gamma
        this.tau = tau
        this.rewardScale = rewardScale

        this._frameInput = tf.input({batchShape : [batchSize, ...this.shape.slice(0, 2), shape[2] * nFrames]})

        this.q1 = this._getCritic("Q1")
        this.q1Optimizer = tf.train.adam()

        this.q1Targ = this._getCritic("Q1_target")
        this.q1Targ.trainable = false

        this.q2 = this._getCritic("Q2")
        this.q2Optimizer = tf.train.adam()
        
        this.q2Targ = this._getCritic("Q2_target")
        this.q2Targ.trainable = false

        this.actor = this._getActor()
        this.actorOptimizer = tf.train.adam()

        this.updateTargets(1)
    }

    /**
     *  Soft update target Q-networks.
     * 
     * @param {number} tau - interpolation factor in polyak averaging: `wTarg <- wTarg*(1-tau) + w*tau`
     */
    updateTargets(tau) {
        tau = tf.scalar(tau)

        const q1W = this.q1.getWeights(),
            q2W = this.q2.getWeights(),
            q1WTarg = this.q1Targ.getWeights(),
            q2WTarg = this.q2Targ.getWeights(),
            len = q1W.length

        const calc = (w, wTarg) => wTarg.mul(tf.scalar(1).sub(tau)).add(w.mul(tau))
        const w1 = [], w2 = []
        for (let i = 0; i < len; i++) {
            w1.push(calc(q1W[i], q1WTarg[i]))
            w2.push(calc(q2W[i], q2WTarg[i]))
        }

        this.q1Targ.setWeights(w1)
        this.q2Targ.setWeights(w2)
    }

    /**
     * Returns actions sampled from normal distribution using means and sigmas predicterd by the actor.
     * 
     * @param {Tensor} state - state
     * @returns {Tensor[]} actions and log expression for loss function
     */
    sampleAction(state) {
        let [mu, sigma] = this.actor.predict(state)
        sigma = tf.clipByValue(sigma, this.noise, 1) // do we need to clip sigma???
  
        // sample epsilon N(mu = 0, sigma = 1)
        const epsilon = tf.randomNormal(mu.shape.slice(0, 2), 0, 1.0)

        // reparameterization trick: z = mu + sigma * epsilon
        const reparamSample = mu.add(sigma.mul(epsilon))

        const action = tf.tanh(reparamSample) // * 1 max_action is 1?
        const logProb = this._logProb(reparamSample, mu, sigma)
  
        const logExpr = logProb.sub(
          tf.log(
            tf.scalar(1)
              .sub(action.pow(tf.scalar(2).toInt()))
              .add(this.noise)
          )
        ).sum(1, true)

        return [action, logProb]
    }

    /**
     * Calculates log probability of normal distribution https://en.wikipedia.org/wiki/Log_probability.
     * Converted to js from https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/distributions/normal.py#L183
     * 
     * @param {Tensor} x - sample from normal distribution with mean `mu` and std `sigma`
     * @param {Tensor} mu - mean
     * @param {Tensor} sigma - standart deviation
     * @returns Log probability
     */
    _logProb(x, mu, sigma)  {
        // sigma = tf.convert_to_tensor(sigma) // already tensor???
        const logUnnormalized = tf.scalar(-0.5).mul(
            tf.squaredDifference(x.div(sigma), mu.div(sigma))
        )
        const logNormalization = tf.scalar(0.5 * Math.log(2 * Math.PI)).add(tf.log(sigma))
    
        return logUnnormalized.sub(logNormalization)
    }

    _getActor(name = "actor") {
        let outputs = this._getConvEncoder(this._frameInput)
        outputs = tf.layers.flatten().apply(outputs)
        outputs = tf.layers.dense({units: 128, activation: "relu"}).apply(outputs)
        outputs = tf.layers.dense({units: 64 , activation: "relu"}).apply(outputs)

        const mu =    tf.layers.dense({units: this.nActions}).apply(outputs)
        const sigma = tf.layers.dense({units: this.nActions}).apply(outputs)

        const model = tf.model({inputs: this._frameInput, outputs: [mu, sigma], name})

        console.log("==========================")
        console.log("==========================")
        console.log("Actor " + name + ": ")

        model.summary()

        return model
    }

    /**
     * Builds critic network model.
     * 
     * @param {string} name - name of the model
     * @returns model
     */
    _getCritic(name = "critic") {
        let outputs = this._getConvEncoder(this._frameInput)

        outputs = tf.layers.flatten().apply(outputs)

        outputs = tf.layers.dense({units: 128, activation: "relu"}).apply(outputs)
        outputs = tf.layers.dense({units: 64 , activation: "relu"}).apply(outputs)

        outputs = tf.layers.dense({units: 1}).apply(outputs)

        const model = tf.model({inputs: this._frameInput, outputs, name})

        console.log("==========================")
        console.log("==========================")
        console.log("CRITIC " + name + ": ")

        model.summary()

        return model
    }

    /**
     * Builds convolutional part of networks.
     * 
     * @param {Tensor} inputs - input for the conv layers
     * @returns outputs
     */
    _getConvEncoder(inputs) {
        const kernelSize = [3, 3]
        const poolSize = 2
        const strides = 1
        const padding = 'same'
        const layers = 14
        
        let filterPow = 2
        let outputs = inputs
        
        for (let i = 0; i < layers; i++) {
            if (i%3 == 1) 
                filterPow++
    
            outputs = tf.layers.conv2d({
                filters: 2**filterPow,
                kernelSize,
                strides,
                padding,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                biasInitializer: 'heNormal'
            }).apply(outputs)
        
            if (i%2 == 1) 
                outputs = tf.layers.maxPooling2d({poolSize}).apply(outputs)
        }
    
        return outputs
    }

    save() {}
    load() {}
    learn() {}
}