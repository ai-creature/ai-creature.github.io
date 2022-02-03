/**
 * Soft Actor Critic Agent https://arxiv.org/abs/1812.05905
 * without value network.
 */
const AgentSac = (() => {
    /**
     * Validates the shape of a given tensor. 
     * 
     * @param {Tensor} tensor - tensor whose shape must be validated
     * @param {array} shape - shape to compare with
     * @param {string} [msg = ''] - message for the error
     */
    const assertShape = (tensor, shape, msg = '') => {
        console.assert(
            JSON.stringify(tensor.shape) === JSON.stringify(shape),
            msg + ' shape ' + tensor.shape + ' is not ' + shape)
    }

    const print = (...args) => console.log(...args)

    // const VERSION = 1 // +100 for bump tower
    // const VERSION = 2 // balls
    // const VERSION = 3 // tests
    // const VERSION = 4 // tests
    const VERSION = 5 // tests

    const LOG_STD_MIN = -20
    const LOG_STD_MAX = 2
    const EPSILON = 1e-8
    const NAME = {
        ACTOR: 'actor',
        Q1: 'q1',
        Q2: 'q2',        
        Q1_TARGET: 'q1-target',
        Q2_TARGET: 'q2-target',
        ALPHA: 'alpha'
    }

    return class AgentSac {
        constructor({
            batchSize = 1, 
            frameShape = [64, 128, 1], 
            nFrames = 2, // Number of stacked frames per state
            nActions = 3, // 3 - impuls, 3 - RGB color
            nTelemetry = 10, // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
            gamma = 0.99, // Discount factor (γ)
            tau = 5e-3, // Target smoothing coefficient (τ)
            trainable = true, // Whether the actor is trainable
            verbose = false,
            forced = false, // force to create fresh models (not from checkpoint)
            prefix = '', // for tests,
            sighted = true
        } = {}) {
            this._batchSize = batchSize
            this._frameShape = frameShape 
            this._nFrames = nFrames
            this._nActions = nActions
            this._nTelemetry = nTelemetry
            this._gamma = gamma
            this._tau = tau
            this._trainable = trainable
            this._verbose = verbose
            this._inited = false
            this._prefix = (prefix === '' ? '' : prefix + '-')
            this._forced = forced
            this._sighted = sighted
            
            this._frameStackShape = [...this._frameShape.slice(0, 2), this._frameShape[2] * this._nFrames]

            // https://github.com/rail-berkeley/softlearning/blob/13cf187cc93d90f7c217ea2845067491c3c65464/softlearning/algorithms/sac.py#L37
            this._targetEntropy = -nActions
        }

        /**
         * Initialization.
         */
        async init() {
            if (this._inited) throw Error('щ（ﾟДﾟщ）')

            this._frameInput = tf.input({batchShape : [null, ...this._frameStackShape]})
            this._telemetryInput = tf.input({batchShape : [null, this._nTelemetry]})
            
            this.actor = await this._getActor(this._prefix + NAME.ACTOR, this.trainable)
            
            if (!this._trainable)
                return
            
            this.actorOptimizer = tf.train.adam()

            this._actionInput = tf.input({batchShape : [null, this._nActions]})

            this.q1 = await this._getCritic(this._prefix + NAME.Q1)
            this.q1Optimizer = tf.train.adam()

            this.q2 = await this._getCritic(this._prefix + NAME.Q2)
            this.q2Optimizer = tf.train.adam()

            this.q1Targ = await this._getCritic(this._prefix + NAME.Q1_TARGET, false) // true for batch norm
            this.q2Targ = await this._getCritic(this._prefix + NAME.Q2_TARGET, false)

            this._logAlpha = await this._getAlpha(this._prefix + NAME.ALPHA)
            this.alphaOptimizer = tf.train.adam()

            this.updateTargets(1)

            this._inited = true
        }

        /**
         * Trains networks on a batch from the replay buffer.
         * 
         * @param {{ state, action, reward, nextState }} - trnsitions in batch
         * @returns {void} nothing
         */
        train({ state, action, reward, nextState }) {
            if (!this._trainable)
                throw new Error('Actor is not trainable')

            return tf.tidy(() => {
                assertShape(state[0], [this._batchSize, this._nTelemetry], 'telemetry')
                assertShape(state[1], [this._batchSize, ...this._frameStackShape], 'frames')
                assertShape(action, [this._batchSize, this._nActions], 'action')
                assertShape(reward, [this._batchSize, 1], 'reward')
                assertShape(nextState[0], [this._batchSize, this._nTelemetry], 'nextState telemetry')
                assertShape(nextState[1], [this._batchSize, ...this._frameStackShape], 'nextState frames')

                this._trainCritics({ state, action, reward, nextState })
                this._trainActor(state)
                this._trainAlpha(state)
                
                this.updateTargets()
            })
        }

        /**
         * Train Q-networks.
         * 
         * @param {{ state, action, reward, nextState }} transition - transition
         */
        _trainCritics({ state, action, reward, nextState }) {
            const getQLossFunction = (() => {
                const [nextFreshAction, logPi] = this.sampleAction(nextState, true)

                const q1TargValue = this.q1Targ.predict(
                    this._sighted ? [...nextState, nextFreshAction] : [nextState[0], nextFreshAction], 
                    {batchSize: this._batchSize})
                const q2TargValue = this.q2Targ.predict(
                    this._sighted ? [...nextState, nextFreshAction] : [nextState[0], nextFreshAction], 
                    {batchSize: this._batchSize})
                
                const qTargValue = tf.minimum(q1TargValue, q2TargValue)
    
                // y = r + γ*(1 - d)*(min(Q1Targ(s', a'), Q2Targ(s', a')) - α*log(π(s'))
                const alpha = tf.exp(this._logAlpha)
                const target = reward.add(
                    tf.scalar(this._gamma).mul(
                        qTargValue.sub(alpha.mul(logPi))
                    )
                )
                            
                assertShape(nextFreshAction, [this._batchSize, this._nActions], 'nextFreshAction')
                assertShape(logPi, [this._batchSize, 1], 'logPi')
                assertShape(qTargValue, [this._batchSize, 1], 'qTargValue')
                assertShape(target, [this._batchSize, 1], 'target')
    
                return (q) => () => {
                    const qValue = q.predict(
                        this._sighted ? [...state, action] : [state[0], action],
                        {batchSize: this._batchSize})
                    
                    // const loss = tf.scalar(0.5).mul(tf.losses.meanSquaredError(qValue, target))
                    const loss = tf.scalar(0.5).mul(tf.mean(qValue.sub(target).square()))
                    
                    assertShape(qValue, [this._batchSize, 1], 'qValue')

                    return loss
                }
            })()
    
            for (const [q, optimizer] of [
                [this.q1, this.q1Optimizer],
                [this.q2, this.q2Optimizer]
            ]) {
                const qLossFunction = getQLossFunction(q)
    
                const { value, grads } = tf.variableGrads(qLossFunction, q.getWeights(true)) // true means trainableOnly
                
                optimizer.applyGradients(grads)
                
                if (this._verbose) console.log(q.name + ' Loss: ' + value.arraySync())
            }
        }

        /**
         * Train actor networks.
         * 
         * @param {state} state 
         */
        _trainActor(state) {
            // TODO: consider delayed update of policy and targets (if possible)
            const actorLossFunction = () => {
                const [freshAction, logPi] = this.sampleAction(state, true)
                
                const q1Value = this.q1.predict(
                    this._sighted ? [...state, freshAction] : [state[0], freshAction],
                    {batchSize: this._batchSize})
                const q2Value = this.q2.predict(
                    this._sighted ? [...state, freshAction] : [state[0], freshAction], 
                    {batchSize: this._batchSize})
                
                const criticValue = tf.minimum(q1Value, q2Value)

                const alpha = tf.exp(this._logAlpha)
                const loss = alpha.mul(logPi).sub(criticValue)

                assertShape(freshAction, [this._batchSize, this._nActions], 'freshAction')
                assertShape(logPi, [this._batchSize, 1], 'logPi')
                assertShape(q1Value, [this._batchSize, 1], 'q1Value')
                assertShape(criticValue, [this._batchSize, 1], 'criticValue')
                assertShape(loss, [this._batchSize, 1], 'alpha loss')

                return tf.mean(loss)
            }
            
            const { value, grads } = tf.variableGrads(actorLossFunction, this.actor.getWeights(true)) // true means trainableOnly
            
            this.actorOptimizer.applyGradients(grads)

            if (this._verbose) console.log('Actor Loss: ' + value.arraySync())
        }

        _trainAlpha(state) {
            const alphaLossFunction = () => {
                const [, logPi] = this.sampleAction(state, true)

                const alpha = tf.exp(this._logAlpha)
                const loss = tf.scalar(-1).mul(
                    alpha.mul( // TODO: not sure whether this should be alpha or logAlpha
                        logPi.add(tf.scalar(this._targetEntropy))
                    )
                )

                assertShape(loss, [this._batchSize, 1], 'alpha loss')

                return tf.mean(loss)
            }
            
            const { value, grads } = tf.variableGrads(alphaLossFunction, [this._logAlpha]) // true means trainableOnly
            
            this.alphaOptimizer.applyGradients(grads)
            
            if (this._verbose) console.log('Alpha Loss: ' + value.arraySync(), tf.exp(this._logAlpha).arraySync())
        }

        /**
         * Soft update target Q-networks.
         * 
         * @param {number} [tau = this._tau] - smoothing constant τ for exponentially moving average: `wTarg <- wTarg*(1-tau) + w*tau`
         */
        updateTargets(tau = this._tau) {
            tau = tf.scalar(tau)

            const
                q1W = this.q1.getWeights(),
                q2W = this.q2.getWeights(),
                q1WTarg = this.q1Targ.getWeights(),
                q2WTarg = this.q2Targ.getWeights(),
                len = q1W.length
            // print(q1W, q1WTarg)
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
         * Returns actions sampled from normal distribution using means and stds predicted by the actor.
         * 
         * @param {Tensor[]} state - state
         * @param {Tensor} [withLogProbs = false] - whether return log probabilities
         * @returns {Tensor || Tensor[]} action and log policy
         */
        sampleAction(state, withLogProbs = false) { // timer ~3ms
            return tf.tidy(() => {
                let [ mu, logStd ] = this.actor.predict(this._sighted ? state : state[0], {batchSize: this._batchSize})

                // https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/torch/sac/policies/gaussian_policy.py#L106
                logStd = tf.clipByValue(logStd, LOG_STD_MIN, LOG_STD_MAX) 
                
                const std = tf.exp(logStd)

                // sample normal N(mu = 0, std = 1)
                const normal = tf.randomNormal(mu.shape, 0, 1.0)
        
                // reparameterization trick: z = mu + std * epsilon
                let pi = mu.add(std.mul(normal))

                let logPi = this._gaussianLikelihood(pi, mu, logStd)

                ;({ pi, logPi } = this._applySquashing(pi, mu, logPi))

                if (!withLogProbs)
                    return pi
        
                return [pi, logPi]
            })
        }

        /**
         * Calculates log probability of normal distribution https://en.wikipedia.org/wiki/Log_probability.
         * Converted to js from https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/distributions/normal.py#L183
         * 
         * @param {Tensor} x - sample from normal distribution with mean `mu` and std `std`
         * @param {Tensor} mu - mean
         * @param {Tensor} std - standart deviation
         * @returns {Tensor} log probability
         */
        _logProb(x, mu, std)  {
            const logUnnormalized = tf.scalar(-0.5).mul(
                tf.squaredDifference(x.div(std), mu.div(std))
            )
            const logNormalization = tf.scalar(0.5 * Math.log(2 * Math.PI)).add(tf.log(std))
        
            return logUnnormalized.sub(logNormalization)
        }

        /**
         * Gaussian likelihood.
         * Translated from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L24
         * 
         * @param {Tensor} x - sample from normal distribution with mean `mu` and std `exp(logStd)`
         * @param {Tensor} mu - mean
         * @param {Tensor} logStd - log of standart deviation
         * @returns {Tensor} log probability
         */
        _gaussianLikelihood(x, mu, logStd) {
            // pre_sum = -0.5 * (
            //     ((x-mu)/(tf.exp(log_std)+EPS))**2 
            //     + 2*log_std 
            //     + np.log(2*np.pi)
            // )

            const preSum = tf.scalar(-0.5).mul(
                x.sub(mu).div(
                    tf.exp(logStd).add(tf.scalar(EPSILON))
                ).square()
                .add(tf.scalar(2).mul(logStd))
                .add(tf.scalar(Math.log(2 * Math.PI)))
            )

            return tf.sum(preSum, 1, true)
        }

        /**
         * Adjustment to log probability when squashing action with tanh
         * Enforcing Action Bounds formula derivation https://stats.stackexchange.com/questions/239588/derivation-of-change-of-variables-of-a-probability-density-function
         * Translated from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L48
         * 
         * @param {*} pi - policy sample
         * @param {*} mu - mean
         * @param {*} logPi - log probability
         * @returns {{ pi, mu, logPi }} squashed and adjasted input
         */
        _applySquashing(pi, mu, logPi) {
            // logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

            const adj = tf.scalar(2).mul(
                tf.scalar(Math.log(2))
                .sub(pi)
                .sub(tf.softplus(
                    tf.scalar(-2).mul(pi)
                ))
            )

            logPi = logPi.sub(tf.sum(adj, 1, true))
            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            return { pi, mu, logPi }
        }

        /**
         * Builds actor network model.
         * 
         * @param {string} [name = 'actor'] - name of the model
         * @param {string} trainable - whether a critic is trainable
         * @returns {tf.LayersModel} model
         */
        async _getActor(name = 'actor', trainable = true) {
            const checkpoint = await this._loadCheckpoint(name)
            if (checkpoint) return checkpoint

            let outputs = this._telemetryInput
            outputs = tf.layers.dense({units: 64, activation: 'relu'}).apply(outputs)

            if (this._sighted) {
                let convOutput = this._getConvEncoder(this._frameInput)
                // convOutput = tf.layers.dense({units: 256, activation: 'relu'}).apply(convOutput)

                outputs = tf.layers.concatenate().apply([convOutput, outputs])
            }

            // outputs = tf.layers.dense({units: 512, activation: 'relu'}).apply(outputs)
            outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs)

            const mu     = tf.layers.dense({units: this._nActions}).apply(outputs)
            const logStd = tf.layers.dense({units: this._nActions}).apply(outputs)

            const model = tf.model({inputs: this._sighted ? [this._telemetryInput, this._frameInput] : [this._telemetryInput], outputs: [mu, logStd], name})
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
        async _getCritic(name = 'critic', trainable = true) {
            const checkpoint = await this._loadCheckpoint(name)
            if (checkpoint) return checkpoint

            let outputs = tf.layers.concatenate().apply([this._telemetryInput, this._actionInput])
            outputs = tf.layers.dense({units: 64, activation: 'relu'}).apply(outputs)

            if (this._sighted) {
                let convOutput = this._getConvEncoder(this._frameInput)
                // convOutput = tf.layers.dense({units: 256, activation: 'relu'}).apply(convOutput)

                outputs = tf.layers.concatenate().apply([convOutput, outputs])
            }

            // outputs = tf.layers.dense({units: 512, activation: 'relu'}).apply(outputs)
            outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs)

            outputs = tf.layers.dense({units: 1}).apply(outputs)

            const model = tf.model({
                inputs: this._sighted 
                    ? [this._telemetryInput, this._frameInput,  this._actionInput] 
                    : [this._telemetryInput, this._actionInput],
                outputs, name
            })

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
            const kernelSize = 3
            const padding = 'valid'
            const poolSize = 2
            const strides = 1
            // const depthwiseInitializer = 'heNormal'
            // const pointwiseInitializer = 'heNormal'
            const kernelInitializer = 'glorotNormal'
            const biasInitializer = 'glorotNormal'

            let outputs = inputs
            
            outputs = tf.layers.conv2d({
                filters: 32,
                kernelSize: 8,
                strides: 4,
                padding,
                kernelInitializer,
                biasInitializer,
                activation: 'relu'
            }).apply(outputs)
            // outputs = tf.layers.maxPooling2d({poolSize}).apply(outputs)
            
            // outputs = tf.layers.layerNormalization().apply(outputs)

            outputs = tf.layers.conv2d({
                filters: 64,
                kernelSize: 4,
                strides: 2,
                padding,
                kernelInitializer,
                biasInitializer,
                activation: 'relu'
            }).apply(outputs)
            // outputs = tf.layers.maxPooling2d({poolSize}).apply(outputs)

            // outputs = tf.layers.layerNormalization().apply(outputs)
            
            outputs = tf.layers.conv2d({
                filters: 64,
                kernelSize: 3,
                strides: 1,
                padding,
                kernelInitializer,
                biasInitializer,
                activation: 'relu'
            }).apply(outputs)
            // outputs = tf.layers.maxPooling2d({poolSize}).apply(outputs)
          
            outputs = tf.layers.conv2d({
                filters: 64,
                kernelSize: 4,
                strides: 1,
                padding,
                kernelInitializer,
                biasInitializer,
                activation: 'relu'
            }).apply(outputs)

            // outputs = tf.layers.batchNormalization().apply(outputs)

            // outputs = tf.layers.layerNormalization().apply(outputs)

            outputs = tf.layers.flatten().apply(outputs)

            // convOutputs = tf.layers.dense({units: 96, activation: 'relu'}).apply(convOutputs)

            return outputs
        }

        /**
         * Builds a log of entropy scale (α) for training.
         * 
         * @param {string} name 
         * @returns {tf.Variable} trainable variable for log entropy
         */
         async _getAlpha(name = 'alpha') {
            let logAlpha = 0.0

            const checkpoint = await this._loadCheckpoint(name)
            if (checkpoint) {
                logAlpha = checkpoint.getWeights()[0].arraySync()[0][0]

                if (this._verbose)
                    print('Checkpoint alpha: ', logAlpha)
                    
                this._logAlphaPlaceholder = checkpoint
            } else {
                const model = tf.sequential({ name });
                model.add(tf.layers.dense({ units: 1, inputShape: [1], useBias: false }))
                model.setWeights([tf.tensor([logAlpha], [1, 1])])

                this._logAlphaPlaceholder = model
            }

            return tf.variable(tf.scalar(logAlpha), true) // true -> trainable
        }

        /**
         * Saves all agent's models to the storage.
         */
        async checkpoint() {
            if (!this._trainable) throw new Error('(╭ರ_ ⊙ )')

            this._logAlphaPlaceholder.setWeights([tf.tensor([this._logAlpha.arraySync()], [1, 1])])

            await Promise.all([
                this._saveCheckpoint(this.actor),
                this._saveCheckpoint(this.q1),
                this._saveCheckpoint(this.q2),
                this._saveCheckpoint(this.q1Targ),
                this._saveCheckpoint(this.q2Targ),
                this._saveCheckpoint(this._logAlphaPlaceholder)
            ])

            if (this._verbose) 
                print('Checkpoint succesfully saved')
        }

        /**
         * Saves a model to the storage.
         * 
         * @param {tf.LayersModel} model 
         */
        async _saveCheckpoint(model) {
            const key = this._getChKey(model.name)
            const saveResults = await model.save(key)

            if (this._verbose) 
                print('Checkpoint saveResults', model.name, saveResults)
        }

        /**
         * Loads saved checkpoint from the storage.
         * 
         * @param {string} name model name
         * @returns {tf.LayersModel} model
         */
        async _loadCheckpoint(name) {
// return
            if (this._forced) {
                print('Forced to not load from the checkpoint ' + name)
                return
            }

            const key = this._getChKey(name)
            const modelsInfo = await tf.io.listModels()

            if (key in modelsInfo) {
                const model = await tf.loadLayersModel(key)

                if (this._verbose) 
                    print('Loaded checkpoint for ' + name)

                return model
            }
            
            if (this._verbose) 
                print('Checkpoint not found for ' + name)
        }
        
        /**
         * Builds the key for the model weights in LocalStorage.
         * 
         * @param {tf.LayersModel} name model name
         * @returns {string} key
         */
        _getChKey(name) {
            return 'indexeddb://' + name + '-' + VERSION
        }
    }
})()

/* TESTS */
;(async () => {
    return 

    const print = (...args) => console.log(...args)

    // https://www.wolframalpha.com/input/?i2d=true&i=y%5C%2840%29x%5C%2844%29+%CE%BC%5C%2844%29+%CF%83%5C%2841%29+%3D+ln%5C%2840%29Divide%5B1%2CSqrt%5B2*%CF%80*Power%5B%CF%83%2C2%5D%5D%5D*Exp%5B-Divide%5B1%2C2%5D*%5C%2840%29Divide%5BPower%5B%5C%2840%29x-%CE%BC%5C%2841%29%2C2%5D%2CPower%5B%CF%83%2C2%5D%5D%5C%2841%29%5D%5C%2841%29
    ;(() => {
        const agent = new AgentSac()

        const 
            mu = tf.tensor([0], [1, 1]),     // mu = 0
            logStd = tf.tensor([0], [1, 1]), // logStd = 0
            std = tf.exp(logStd),            // std = 1
            normal = tf.tensor([0], [1, 1]), // N = 0
            pi = mu.add(std.mul(normal))     // x = 0
    
        const log = agent._gaussianLikelihood(pi, mu, logStd)

        console.assert(log.arraySync()[0][0].toFixed(5) === '-0.91894', 
            'test Gaussian Likelihood for μ=0, σ=1, x=0')
    })()

    ;(() => {
        const agent = new AgentSac()

        const 
            mu = tf.tensor([1], [1, 1]),     // mu = 1
            logStd = tf.tensor([1], [1, 1]), // logStd = 1
            std = tf.exp(logStd),            // std = e
            normal = tf.tensor([0], [1, 1]), // N = 0
            pi = mu.add(std.mul(normal))    // x = 1
    
        const log = agent._gaussianLikelihood(pi, mu, logStd)

        console.assert(log.arraySync()[0][0].toFixed(5) === '-1.91894',
            'test Gaussian Likelihood for μ=1, σ=e, x=0')
    })()

    ;(() => {
        const agent = new AgentSac()

        const 
            mu = tf.tensor([1], [1, 1]),     // mu = -1
            logStd = tf.tensor([1], [1, 1]), // logStd = 1
            std = tf.exp(logStd),            // std = e
            normal = tf.tensor([0.1], [1, 1]), // N = 0
            pi = mu.add(std.mul(normal))    // x = -1.27182818
    
        const logPi = agent._gaussianLikelihood(pi, mu, logStd)
        const { pi: piSquashed, logPi: logPiSquashed } = agent._applySquashing(pi, mu, logPi)

        const logProbBounded = logPi.sub(
          tf.log(
            tf.scalar(1)
              .sub(tf.tanh(pi).pow(tf.scalar(2)))
              // .add(EPSILON)
          )
        ).sum(1, true)
        
        console.assert(logPi.arraySync()[0][0].toFixed(5) === '-1.92394',
            'test Gaussian Likelihood for μ=-1, σ=e, x=-1.27182818')

        console.assert(logPiSquashed.arraySync()[0][0].toFixed(5) === logProbBounded.arraySync()[0][0].toFixed(5),
            'test logPiSquashed for μ=-1, σ=e, x=-1.27182818')

        console.assert(piSquashed.arraySync()[0][0].toFixed(5) === tf.tanh(pi).arraySync()[0][0].toFixed(5),
            'test piSquashed for μ=-1, σ=e, x=-1.27182818')
    })()

    await (async () => {
        const state = tf.tensor([
            0.5, 0.3, -0.9,
            0, -0.8, 1,
            -0.3, 0.04, 0.02,
            0.9
        ], [1, 10])

        const action = tf.tensor([
            0.1, -1, -0.4,
            1, -0.8, -0.8, -0.2,
            0.04, 0.02, 0.001
        ], [1, 10])
        
        const fresh = new AgentSac({ prefix: 'test', forced: true })
        await fresh.init()
        await fresh.checkpoint()
        
        const saved = new AgentSac({ prefix: 'test' })
        await saved.init()
        
        let frPred, saPred

        frPred = fresh.actor.predict(state, {batchSize: 1})
        saPred = saved.actor.predict(state, {batchSize: 1})
        console.assert(
            frPred[0].arraySync().length > 0 &&
            frPred[1].arraySync().length > 0 &&
            frPred[0].arraySync().join(';') === saPred[0].arraySync().join(';') &&
            frPred[1].arraySync().join(';') === saPred[1].arraySync().join(';'),
            'Models loaded from the checkpoint should be the same')
        
        frPred = fresh.q1.predict([state, action], {batchSize: 1})
        saPred = fresh.q1Targ.predict([state, action], {batchSize: 1})
        console.assert(
            frPred.arraySync()[0][0] !== undefined &&
            frPred.arraySync()[0][0] === saPred.arraySync()[0][0],
            'Q1 and Q1-target should be the same')

        frPred = fresh.q2.predict([state, action], {batchSize: 1})
        saPred = saved.q2.predict([state, action], {batchSize: 1})
        console.assert(
            frPred.arraySync()[0][0] !== undefined &&
            frPred.arraySync()[0][0] === saPred.arraySync()[0][0],
            'Q and Q restored should be the same')

        console.assert(
            fresh._logAlpha.arraySync() !== undefined &&
            fresh._logAlpha.arraySync() === fresh._logAlpha.arraySync(),
            'Q and Q restored should be the same')
    })()
})()
