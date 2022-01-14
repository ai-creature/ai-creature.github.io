/**
 * Validates the shape of a given tensor. 
 * 
 * @param {Tensor} tensor - tensor whose shape must be validated
 * @param {array} shape - shape to compare with
 * @param {string} [msg = ''] - message for the error
 */
const assertShape = (tensor, shape, msg = '') => {
    tf.util.assert(
        JSON.stringify(tensor.shape) === JSON.stringify(shape),
        msg + ' shape ' + tensor.shape + ' is not ' + shape)
}

/**
 * Soft Actor Critic Agent https://arxiv.org/abs/1812.05905
 * without value network.
 */
class AgentSac {
    constructor({
        batchSize = 1, 
        frameShape = [128, 256, 3], 
        nFrames = 4, // Number of stacked frames per state
        nActions = 9, // 3 impulses by axis, 3 rotations, rgb color
        nTelemetry = 6, // 3 angular speeds and 3 speeds by axis
        epsilon = 1e-6, // Small number
        alpha = 0.5, // Entropy scale (α)
        gamma = 0.99, // Discount factor (γ)
        tau = 5e-3, // Target smoothing coefficient (τ)
        rewardScale = 2,
        trainable = true, // Whether the actor is trainable
        verbose = false
    }) {
        this._batchSize = batchSize
        this._frameShape = frameShape 
        this._nFrames = nFrames
        this._nActions = nActions
        this._nTelemetry = nTelemetry
        this._epsilon = epsilon
        this._alpha = alpha
        this._gamma = gamma
        this._tau = tau
        this._rewardScale = rewardScale
        this._trainable = trainable
        this._verbose = verbose

        this._frameInput = tf.input({batchShape : [null, ...frameShape.slice(0, 2), frameShape[2] * nFrames]})
        this._telemetryInput = tf.input({batchShape : [null, nTelemetry]})
        this._actionInput = tf.input({batchShape : [null, nActions]})

        this.actor = this._getActor('Actor', trainable)

        if (this._trainable) {
            this.actorOptimizer = tf.train.adam()

            this.q1 = this._getCritic('Q1')
            this.q1Optimizer = tf.train.adam()

            this.q1Targ = this._getCritic('Q1_target', false)

            this.q2 = this._getCritic('Q2')
            this.q2Optimizer = tf.train.adam()
            
            this.q2Targ = this._getCritic('Q2_target', false)

            this.updateTargets(1)
        }
    }

    /**
     * Trains networks on a batch from the replay buffer.
     * 
     * @param {*} st 
     */
    learn(st) {
        if (!this._trainable) {
            throw new Error('Actor is not trainable')
        }

        const 
            state = st,
            action = tf.ones([this._batchSize, this._nActions]),
            reward = tf.ones([this._batchSize, 1]),
            nextState = tf.onesLike(state)

        assertShape(state, [this._batchSize, ...this._frameShape], 'state')
        assertShape(action, [this._batchSize, this._nActions], 'action')
        assertShape(reward, [this._batchSize, 1], 'reward')
        assertShape(nextState, state.shape, 'nextState')

        const getQLossFunction = (() => {
            const [nextFreshAction, logProb] = this.sampleAction(nextState, true)
            
            const q1TargValue = this.q1Targ.predict([nextState, nextFreshAction], {batchSize: this._batchSize})
            const q2TargValue = this.q2Targ.predict([nextState, nextFreshAction], {batchSize: this._batchSize})
            
            const qTargValue = tf.minimum(q1TargValue, q2TargValue)

            // y = r + γ*(1 - d)*(min(Q1Targ(s', a'), Q2Targ(s', a')) - α*log(π(s'))
            const target = reward.mul(tf.scalar(this._rewardScale))
                .add(
                    tf.scalar(this._gamma).mul(
                        qTargValue.sub(tf.scalar(this._alpha).mul(logProb))
                    )
                )
                        
            assertShape(nextFreshAction, [this._batchSize, this._nActions], 'nextFreshAction')
            assertShape(logProb, [this._batchSize, 1], 'logProb')
            assertShape(qTargValue, [this._batchSize, 1], 'qTargValue')
            assertShape(target, [this._batchSize, 1], 'target')

            return (q) => {
                return () => {
                    const qValue = q.predict([state, action], {batchSize: this._batchSize})
                    
                    const loss = tf.scalar(0.5).mul(tf.losses.meanSquaredError(qValue, target))
                    
                    assertShape(qValue, [this._batchSize, 1], 'qValue')

                    return loss
                }
            }
        })()

        for (const [q, optimizer] of [[this.q1, this.q1Optimizer], [this.q2, this.q2Optimizer]]) {
            const qLossFunction = getQLossFunction(q)

            const {value, grads} = tf.variableGrads(qLossFunction, q.getWeights(true)) // true means trainableOnly

            optimizer.applyGradients(grads)
        }

        // TODO: consider delayed update of policy and targets (if possible)
        const actorLossFunction = () => {
            const [freshAction, logProb] = this.sampleAction(state, true)
            
            const q1Value = this.q1.predict([state, freshAction], {batchSize: this._batchSize})
            const q2Value = this.q2.predict([state, freshAction], {batchSize: this._batchSize})
            
            const criticValue = tf.minimum(q1Value, q2Value)
            
            const loss = tf.mean(tf.scalar(this._alpha).mul(logProb).sub(criticValue))
            assertShape(freshAction, [this._batchSize, this._nActions], 'freshAction')
            assertShape(logProb, [this._batchSize, 1], 'logProb')
            assertShape(q1Value, [this._batchSize, 1], 'q1Value')
            assertShape(criticValue, [this._batchSize, 1], 'criticValue')
            
            return loss
        }
        
        const {value, grads} = tf.variableGrads(actorLossFunction, this.actor.getWeights(true)) // true means trainableOnly
        
        this.actorOptimizer.applyGradients(grads)
        
        if (this._verbose) console.log('Actor Loss: ' + value)
        
        this.updateTargets()
    }

    /**
     * Soft update target Q-networks.
     * 
     * @param {number} [tau = this._tau] - interpolation factor in polyak averaging: `wTarg <- wTarg*(1-tau) + w*tau`
     */
    updateTargets(tau = this._tau) {
        tau = tf.scalar(tau)

        const
            q1W = this.q1.getWeights(),
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
     * Returns actions sampled from normal distribution using means and sigmas predicted by the actor.
     * 
     * @param {Tensor} state - state
     * @param {Tensor} [withLogProbs = false] - whether return log probabilities
     * @returns {Tensor || Tensor[]} action and log policy
     */
    sampleAction(state, withLogProbs = false) { // timer ~3ms
        return tf.tidy(() => {
            let [mu, sigma] = this.actor.predict(state, {batchSize: this._batchSize})
            sigma = tf.clipByValue(sigma, this._epsilon, 1) // do we need to clip sigma??? 
            // TODO: output log(std) instead of std
            // assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/torch/sac/policies/gaussian_policy.py#L106
      
            // sample epsilon N(mu = 0, sigma = 1)
            const epsilon = tf.randomNormal(mu.shape.slice(0, 2), 0, 1.0)
    
            // reparameterization trick: z = mu + sigma * epsilon
            const reparamSample = mu.add(sigma.mul(epsilon))
    
            const action = tf.tanh(reparamSample) // * 1 max_action is 1?
    
            if (!withLogProbs) return action
    
            const logProb = this._logProb(reparamSample, mu, sigma)
      
            // Enforcing Action Bound
            const logProbBounded = logProb.sub(
              tf.log(
                tf.scalar(1)
                  .sub(action.pow(tf.scalar(2).toInt()))
                  .add(this._epsilon)
              )
            ).sum(1, true) // TODO: figure out why we sum log_probs together with squash_correction
    
            return [action, logProbBounded]
        })
    }

    /**
     * Calculates log probability of normal distribution https://en.wikipedia.org/wiki/Log_probability.
     * Converted to js from https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/distributions/normal.py#L183
     * 
     * @param {Tensor} x - sample from normal distribution with mean `mu` and std `sigma`
     * @param {Tensor} mu - mean
     * @param {Tensor} sigma - standart deviation
     * @returns {Tensor} log probability
     */
    _logProb(x, mu, sigma)  {
        const logUnnormalized = tf.scalar(-0.5).mul(
            tf.squaredDifference(x.div(sigma), mu.div(sigma))
        )
        const logNormalization = tf.scalar(0.5 * Math.log(2 * Math.PI)).add(tf.log(sigma))
    
        return logUnnormalized.sub(logNormalization)
    }

    /**
     * Builds actor network model.
     * 
     * @param {string} [name = 'actor'] - name of the model
     * @param {string} trainable - whether a critic is trainable
     * @returns {tf.LayersModel} model
     */
    _getActor(name = 'actor', trainable = true) {
        let outputs = this._getConvEncoder(this._frameInput)
        outputs = tf.layers.flatten().apply(outputs)
        outputs = tf.layers.dense({units: 128, activation: 'relu'}).apply(outputs)
        outputs = tf.layers.dense({units: 64 , activation: 'relu'}).apply(outputs)

        const mu =    tf.layers.dense({units: this._nActions}).apply(outputs)
        const sigma = tf.layers.dense({units: this._nActions}).apply(outputs)

        const model = tf.model({inputs: this._frameInput, outputs: [mu, sigma], name})
        model.trainable = trainable

        if (this._verbose) {
            console.log('==========================')
            console.log('==========================')
            console.log('Actor ' + name + ': ')

            model.summary()
        }

        return model
    }

    /**
     * Builds a critic network model.
     * 
     * @param {string} [name = 'critic'] - name of the model
     * @param {string} trainable - whether a critic is trainable
     * @returns {tf.LayersModel} model
     */
    _getCritic(name = 'critic', trainable = true) {
        let convOutputs = this._getConvEncoder(this._frameInput)
        convOutputs = tf.layers.flatten().apply(convOutputs)

        const concatOutput = tf.layers.concatenate().apply([convOutputs, this._actionInput])

        let outputs = tf.layers.dense({units: 128, activation: 'relu'}).apply(concatOutput)
        outputs = tf.layers.dense({units: 64 , activation: 'relu'}).apply(outputs)

        outputs = tf.layers.dense({units: 1}).apply(outputs)

        const model = tf.model({inputs: [this._frameInput, this._actionInput], outputs, name})
        model.trainable = trainable

        if (this._verbose) {
            console.log('==========================')
            console.log('==========================')
            console.log('CRITIC ' + name + ': ')
    
            model.summary()
        }

        return model
    }

    /**
     * Builds convolutional part of a network.
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
}