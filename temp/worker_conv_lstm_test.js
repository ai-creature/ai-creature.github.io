importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");
importScripts("my_gru_layer.js")

// import * as tfvis from '@tensorflow/tfjs-vis';

const shape = [128, 256, 3]
const batchSize = 1
const rnnUnits = 32
const kernelSize = [3, 3]
const poolSize = 2
const strides = 1
const padding = 'same'
const layers = 14

const inputs = tf.input({batchShape : [batchSize, ...shape]})

let filterPow = 2
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







/*

const layers = 7

const inputs = tf.input({batchShape : [batchSize, ...shape]})

let filterPow = 4
let outputs = inputs

for (let i = 0; i < layers; i++) {
    if (i%2 == 0) {
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

    // if (i%2 == 1) {
        outputs = tf.layers.maxPooling2d({
            poolSize
        }).apply(outputs)
    // }
}

 */



















// outputs = tf.layers.layerNormalization().apply(outputs)


// outputs = tf.layers.reshape({targetShape: [2, 128]}).apply(outputs)
// outputs = tf.layers.permute({dims: [2,1]}).apply(outputs)

outputs = tf.layers.flatten().apply(outputs)
outputs = tf.layers.repeatVector({n: 2}).apply(outputs)
let rnnState = []
;[outputs, rnnState] = tf.layers.gru({
    units: rnnUnits,
    stateful: true,
    returnSequences: false,
    returnState: true
}).apply(outputs/*, {initialState: inputsInitState}/*)

// outputs = tf.layers.dense({units: 32}).apply(outputs)

// outputs = tf.layers.flatten().apply(outputs)
// outputs = new MyGruLayer({units: rnnUnits}).apply(outputs/*[outputs, inputsInitState]*/)

const model = tf.model({inputs/*: [inputs, inputsInitState]*/, outputs})
const optimizer = tf.train.adam()
// model.compile({
//     optimizer, 
//     loss: 'meanSquaredError',
//     // metrics: ['accuracy']
// })

console.log(model.summary())

let busy = false
let i = 0
let prevIsBlack = false

const SAME = false

// let prevState = tf.randomNormal([batchSize, rnnUnits])

self.addEventListener('message', async e => {
    if (busy) return
    busy = true
    i++

    const meanRgb = [0.485, 0.456, 0.406]
    const stdRgb = [0.229, 0.224, 0.225]

    // const isBlack = Math.random() <= 0.5
    const isBlack = i%3 === 0 // Math.random() <= 0.3
    const val = isBlack ? 0 : 255

    const shift = 50
    const side = 12
    for (let i = shift; i < shift + side; i++) {
        for (let j = shift; j < shift + side; j++) {
            e.data[i][j] = [val, val, val]
        }
    }

    const frame = tf.tensor3d(e.data, shape, 'float32')
    // const frame = isBlack ? tf.fill(shape, 0, 'float32') : tf.fill(shape, 255, 'float32')

    const frameNorm = frame
        .div(tf.scalar(255))
        // .sub(meanRgb)
        // .div(stdRgb)

    if (i%128 === 0) {
        console.log("***")
        model.resetStates()
    }

    const input = tf.stack([frameNorm])

    if (SAME) prevIsBlack = isBlack

    const labels = prevIsBlack ? tf.ones([1, rnnUnits]) : tf.zeros([1, rnnUnits])
    //console.log(val, labels.dataSync())
    // const h = await model.fit(input, labels, {
    //     batchSize,
    //     epochs: 1,
    //     shuffle: false,
    //     verbose: 2
    // })

    
    // console.log("Loss: " + h.history.loss[0].toFixed(4)/*, "Acc: " + h.history.acc[0].toFixed(2)*/)
    
    const lossFunction = () => tf.tidy(() => {
        const preds = model.predict(input)
        // const preds = model.apply(input, {training: true})
        
        return tf.losses.meanSquaredError(labels, preds).asScalar()
    })

    // const grads = optimizer.computeGradients(f)
    const {value, grads} = tf.variableGrads(lossFunction)
    
    // optimizer.applyGradients(tf.zip(model.weights, grads))
    optimizer.applyGradients(grads)
     
    console.log("Loss: " + value)
    
    
    if (!SAME) prevIsBlack = isBlack
    
    
    
    
    
    
    tf.dispose(value)
    tf.dispose(grads)

    input.dispose()
    // res.dispose()
    labels.dispose()

    const data = await frame.array()
    // console.log(data[0][0])

    frame.dispose()
    frameNorm.dispose()

    self.postMessage(data)

    busy = false
})

