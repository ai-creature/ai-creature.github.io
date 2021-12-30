importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");
importScripts("my_gru_layer.js")

const shape = [128, 256, 3]
const batchSize = 1
const kernelSize = [3, 3]
const poolSize = 2
const strides = 1
const padding = 'same'
const layers = 14
const inputs = tf.input({batchShape : [batchSize, ...shape]})

let filterPow = 3
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
        kernelInitializer: 'heNormal',
        biasInitializer: 'heNormal',
        trainable: true
    }).apply(outputs)

    if (i%2 == 1) {
        outputs = tf.layers.maxPooling2d({
            poolSize
        }).apply(outputs)
    }
}

// outputs = tf.layers.reshape({targetShape: [1, ...outputs.shape.slice(1)]}).apply(outputs)
// outputs = tf.layers.convLstm2d({filters: 64, kernelSize}).apply(outputs)
// outputs = tf.layers.maxPooling2d({poolSize}).apply(outputs)


// let rnnState = []
// outputs = tf.layers.flatten().apply(outputs)
// ;[outputs, rnnState] = tf.layers.rnn({
//     cell:[
//         tf.layers.gruCell({
//             units: 32,
//             kernelInitializer: 'heNormal',
//             recurrentInitializer: 'heNormal',
//             // biasInitializer: 'heNormal',
//             // useBias: true,
//             // trainable: true

//             // dropout: 0.1,
//             // recurrentDropout: 0.1,
//             // activation: 'relu'
//         }),
//     ],
//     stateful: true,
//     returnSequences: false,
//     returnState: true
// }).apply(outputs)


outputs = tf.layers.flatten().apply(outputs)
outputs = tf.layers.repeatVector({n: 2}).apply(outputs)
// outputs = tf.layers.reshape({targetShape: [1, 512]}).apply(outputs)
// outputs = tf.layers.permute({dims: [2,1]}).apply(outputs)
outputs = tf.layers.gru({units: 32, stateful: true}).apply(outputs)

// outputs = new MyGruLayer({units: 32}).apply(outputs)

// outputs = tf.layers.dense({units: 64}).apply(outputs)



const model = tf.model({inputs, outputs})
model.compile({optimizer: 'adam', loss: 'meanSquaredError'})
console.log(model.summary())
// model.weights.forEach(w => {console.log(w.name, w.shape);});


let busy = false
let i = 0
let prevIsBlack = false
self.addEventListener('message', async e => {
    if (busy) return

    i++

    const meanRgb = [0.485, 0.456, 0.406]
    const stdRgb = [0.229, 0.224, 0.225]

    // const isBlack = Math.random() <= 0.3
    const isBlack = i%4 === 0 // Math.random() <= 0.3
    const val = isBlack ? 0 : 255

    for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
            e.data[i][j] = [val, val, val]
        }
    }

    const frame = tf.tensor3d(e.data, shape, 'float32')
    // const frame = isBlack ? tf.fill(shape, 0, 'float32') : tf.fill(shape, 255, 'float32')

    const frameNorm = frame
        .div(tf.scalar(255))
        // .sub(meanRgb)
        // .div(stdRgb)
    // console.log((await frameNorm.array())[0][0])

    busy = true
i%11 && model.resetStates()
// Math.random() < 0.08 && model.resetStates()

    const input = tf.stack([frameNorm])
    const res = model.predict(input, {batchSize})
    // console.log("Pedict: ", res.shape)
    const labels = i%3 === 0 ? tf.ones(res.shape) : tf.zeros(res.shape)
    const h = await model.fit(input, labels, {
        batchSize,
        epochs: 1,
        shuffle: false,
        verbose: 2
    })

    prevIsBlack = isBlack
    
    console.log("Losss: " + h.history.loss[0].toFixed(2))


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

