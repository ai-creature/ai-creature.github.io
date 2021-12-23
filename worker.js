/*
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
const ys = tf.tensor2d([1, 3, 5, 7, 8], [5, 1]);

await model.fit(xs, ys, {epochs: 100});
model.predict(tf.tensor2d([6], [1, 1])).print(); 
*/

importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");

const model = tf.sequential();
const inputShape = [128, 256, 3]
const kernelSize = [3, 3]
const poolSize = 2
const strides = 1
const padding = 'same'
let filterPow = 3

model.add(tf.layers.inputLayer({inputShape}))

layers = 14
for (let i = 0; i < layers; i++) {
    if (i%3 == 1) {
        filterPow++
        
    }

    model.add(tf.layers.conv2d({
        filters: 2**filterPow,
        kernelSize,
        activation: 'relu',
        padding,
        strides
    }))

    if (i%2 == 1) {
        model.add(tf.layers.maxPooling2d({
            poolSize
        }))
    }
}

model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})
console.log(model.summary())

let busy = false
self.addEventListener('message', async e => {
    const frame = tf.tensor3d(e.data, inputShape)

    if (true && !busy) {
        let st = Date.now()
        console.time("predict: " + st)
        const res = model.predict(frame.expandDims(0))
        // res.print()
        // console.log(res)
        console.timeEnd("predict: " + st)


        busy = true

        st = Date.now()
        console.time("fit: " + st)
        const h = await model.fit(frame.expandDims(0), tf.ones(res.shape), {
            batchSize: 1,
            epochs: 1
        });
        console.log("Loss: " + h.history.loss[0])
        console.timeEnd("fit: " + st)

        busy = false
    }


    const data = await frame.array()

    frame.dispose()

    self.postMessage(data)
})

