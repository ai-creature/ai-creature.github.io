(async () => {
    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js");
                    
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7, 8], [5, 1]);

    // Train the model using the data.
    await model.fit(xs, ys, {epochs: 100});

    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([6], [1, 1])).print();

    self.addEventListener('message', e => {
        console.log(`Message from main thread ${e.data}`)
        // send a message back to the main thread
        self.postMessage("Hello from worker " + e.data.length)
    })
})()