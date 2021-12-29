importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");

const shape = [128, 256, 3]
const batchSize = 1
const kernelSize = [3, 3]
const poolSize = 2
const strides = 1
const padding = 'same'
let filterPow = 3

const inputs = tf.input({
    // shape, 
    batchShape : [batchSize, ...shape]
})

const inputsRnn = tf.input({ 
    batchShape : [1, 8]
})

layers = 14
let outputs = inputs
for (let i = 0; i < layers; i++) {
    if (i%3 == 1) {
        filterPow++
    }

    outputs = tf.layers.conv2d({
        filters: 2**filterPow,
        kernelSize,
        strides,
        padding,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }).apply(outputs)

    if (i%2 == 1) {
        outputs = tf.layers.maxPooling2d({
            poolSize
        }).apply(outputs)
    }
}

outputs = tf.layers.flatten().apply(outputs)

// outputs = tf.layers.dense({units: 64}).apply(outputs)

let rnnState = []
// ;[outputs, ...rnnState]
// outputs = tf.layers.dropout(0.1).apply(outputs)


outputs = tf.layers.repeatVector({n: 2}).apply(outputs)
outputs = tf.layers.rnn({
    cell:[
        tf.layers.gruCell({units: 32}),
        // tf.layers.gruCell({units: 16}),
        // tf.layers.gruCell({units: 8}),
    ],
    // units: 8,
    stateful: true,
    returnSequences: false,
    returnState: false
}).apply(outputs)

// outputs = tf.layers.dense({units: 4}).apply(outputs)


// console.log("RNN inner state:", rnnState.map(s => s.shape))


const model = tf.model({inputs: inputs, outputs})
model.compile({optimizer: 'adam', loss: 'meanSquaredError'})
console.log(model.summary())
// model.weights.forEach(w => {console.log(w.name, w.shape);});


let busy = false
let i = 0
let prevIsBlack = false
self.addEventListener('message', async e => {
    if (true && busy) return

    i++

    const meanRgb = [0.485, 0.456, 0.406]
    const stdRgb = [0.229, 0.224, 0.225]

    const isBlack = Math.random() <= 0.3

    const val = isBlack ? 0 : 255

    e.data[0][0] = [val, val, val]
    e.data[0][1] = [val, val, val]
    e.data[0][2] = [val, val, val]

    e.data[1][0] = [val, val, val]
    e.data[1][1] = [val, val, val]
    e.data[1][2] = [val, val, val]

    e.data[2][0] = [val, val, val]
    e.data[2][1] = [val, val, val]
    e.data[2][2] = [val, val, val]

    const frame = tf.tensor3d(e.data, shape, 'float32')

    const frameNorm = frame
        .div(tf.scalar(255))
        // .sub(meanRgb)
        // .div(stdRgb)
    // console.log((await frameNorm.array())[0][0])

    busy = true
i%11 && model.resetStates()
    const input = tf.stack([frameNorm])
    const res = model.predict(input, {batchSize})
    const labels = prevIsBlack ? tf.ones(res.shape) : tf.zeros(res.shape)
    const h = await model.fit(input, labels, {
        batchSize,
        epochs: 1,
        shuffle: false,
        verbose: 2
    })

    prevIsBlack = isBlack
    
    console.log("Losss: " + h.history.loss[0])


    input.dispose()
    res.dispose()
    labels.dispose()

    

    /*const optimizer = tf.train.rmsprop(0.01)
    const labels = tf.ones(res.shape)
    optimizer.minimize(() => {
        const pred = model.predict(input, {batchSize})
        const loss = tf.losses.meanSquaredError(labels, pred);
        
        loss.data().then(h => {
            busy = false
            console.log('Loss: ', h[0])
        })
        
        return loss;
    })*/
    
    // else console.log("skip")


    const data = await frame.array()
    // console.log(data[0][0])

    frame.dispose()
    frameNorm.dispose()

    self.postMessage(data)

    busy = false
})

