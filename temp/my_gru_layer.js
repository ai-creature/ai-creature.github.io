class MyGruLayer extends tf.layers.Layer {
    constructor(args) {
        super(args)

        this.cell = new tf.layers.gruCell(args)
        this.states_ = null
        this.keptStates = []
    }

    build(inputShape) {
        this.cell.build(inputShape)
        this.resetStates()
        this.stateSpec = {shape: [null, this.cell.stateSize]}
        this.built = true
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.cell.stateSize]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const training = kwargs == null ? null : kwargs['training']
            const cellCallKwargs = {training}
            const input = inputs[0]
            const initialState= this.states_

            const [outputs, ...states] = this.cell.call([input].concat(initialState), cellCallKwargs)

            this.resetStates(states, training)

            return outputs
        })
    }

    /* Method from https://github.com/tensorflow/tfjs/blob/tfjs-v3.12.0/tfjs-layers/src/layers/recurrent.ts#L562 */
    resetStates(states, training = false) {
        tf.tidy(() => {
            if (this.states_ == null) {
                if (Array.isArray(this.cell.stateSize)) {
                    this.states_ = this.cell.stateSize.map(dim => tf.zeros([batchSize, dim]));
                } else {
                    this.states_ = [tf.zeros([batchSize, this.cell.stateSize])];
                }
            } else if (states == null) {
                // Dispose old state tensors.
                tf.dispose(this.states_);
                // For stateful RNNs, fully dispose kept old states.
                if (this.keptStates != null) {
                    tf.dispose(this.keptStates);
                    this.keptStates = [];
                }
        
                if (Array.isArray(this.cell.stateSize)) {
                    this.states_ = this.cell.stateSize.map(dim => tf.zeros([batchSize, dim]));
                } else {
                    this.states_[0] = tf.zeros([batchSize, this.cell.stateSize]);
                }
            } else {
                if (training === true) {
                  this.keptStates.push(this.states_.slice());
                } else {
                  tf.dispose(this.states_);
                }
        
                for (let index = 0; index < this.states_.length; ++index) {
                  const value = states[index];
                  const dim = Array.isArray(this.cell.stateSize) ?
                      this.cell.stateSize[index] :
                      this.cell.stateSize;
                  const expectedShape = [batchSize, dim];
                  if (value.shape[0] != batchSize || value.shape[1] != dim) {
                    throw new Error(
                        `State ${index} is incompatible with layer ${this.name}: ` +
                        `expected shape=${expectedShape}, received shape=${
                            value.shape}`);
                  }
                  this.states_[index] = value;
                }
            }

            this.states_ = this.states_.map(state => tf.keep(state.clone()));
        })
    }

    static get className() {
        return 'MyGruLayer';
    }
}


tf.serialization.registerClass(MyGruLayer)

// outputs = new MyGruLayer({units: 32}).apply(outputs)