// This example is inspired by https://medium.com/tensorflow/getting-started-with-tensorflow-js-50f6783489b2
// I encourage you to read that short post!

async function main() {
    
    // define a single neuron linear model
    const model = tf.sequential()
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))
    // Mean Squared Error (MSE) is the most common loss function for regression 
    // tasks like this one. We'll use the basic Stochastic Gradient Descent
    // optimizer as well!
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })
    
    // create our training set
    const { x, y } = getRealData(100)
    
    // create tf.tensors from the training data for use during model.fit()
    const xTensor = tf.tensor(x)
    const yTensor = tf.tensor(y)
    
    // train the model! This will train for 100 full passes, or epochs, through
    // our 100 sample datas. Meaning our model will see each sample 100 times.
    await model.fit(xTensor, yTensor, { epochs: 100 })
    
    // don't forget to free the GPU memory now that
    // we're done with our training data
    xTensor.dispose()
    yTensor.dispose()
    
    // Now that we've trained our model, it's time for model inference. Let's
    // use our model to predict the output values of ten random numbers.
    for (let i = 0; i < 10; i++) {
        
        const rand = random(-10, 10)
        const real = realFunction(rand)
        // we use model.predict() followed by .dataSync() to download our predicted
        // data from the GPU. Our data is wrapped in a "batch dimension", so we'll
        // have to grab the first (and only) element to get our prediction as a float.
        // Notice that we pass [rand] into tf.tensor and specify an input shape as [1, 1].
        // This is because model.predict(...) expects data in batches, so we are
        // technically passing rand into predict as a mini-batch of one example.
        const pred = model.predict(tf.tensor([rand], [1, 1])).dataSync()[0]
        
        // let's see how our model did by comparing the predicted value to the
        // real output of realFunction(). Write the results to the DOM.
        if (i == 0) document.getElementById('output').innerText = ''        
        const html = `real: ${real.toFixed(2)}, pred: ${pred.toFixed(2)}, error: ${Math.abs(pred - real).toFixed(2)}<br>`
        document.getElementById('output').innerHTML += html
    }
}

// get a random float between min and max
function random(min, max) {
    return Math.random() * (max - min) + min
}

// this is true function that our neural net is trying to learn
// we use it to generate training data and to compare against
// our model's predictions
function realFunction(x) {
    return 2.0 * x - 1.0
}

// generate training data samples in the form of { x, y } using realFunction()
function getRealData(numSamples) {
    
    const x = [], y = []
    
    for (let i = 0; i < numSamples; i++) {
        // pick a random number between -10 and 10
        const rand = random(-10, 10)
        x.push(rand)
        y.push(realFunction(rand))
    }
    
    return { x, y }
}

// run it!
main()