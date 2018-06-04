const fs = require('fs')

const MAX_LEN = 40
const SKIP = 1
const EPOCHS = 5
const LEARNING_RATE = 0.15

function loadFile(path) {
    return new Promise((resolve, reject) => {
        fs.readFile(path, (err, data) => {
            if (err) reject(err)
            else resolve(data)
        })
    })
}

function getModel(inputShape) {
    const model = tf.sequential()
    model.add(tf.layers.lstm({units: 128, inputShape: inputShape, returnSequences: true, recurrentInitializer: 'glorotNormal', activation: 'relu'}))
    model.add(tf.layers.lstm({units: 128, returnSequences: false, recurrentInitializer: 'glorotNormal', activation: 'relu'}))
    model.add(tf.layers.dense({units: inputShape[1]}))
    model.add(tf.layers.activation({ activation: 'softmax' }))
    model.compile({
        optimizer: tf.train.rmsprop(LEARNING_RATE),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    return model
}

async function getData(path) {
    
    const buf = await loadFile(path)
    const text = buf.toString().toLowerCase().slice(0, 10000)

    console.log(`corpuslength: ${text.length}`)    
    const chars = Array.from(new Set(text))
    console.log(`total chars: ${chars.length}`)
    
    const charIndicies = {}; chars.forEach((c, i) => charIndicies[c] = i)
    
    // cut the text in semi-redundant sequences of maxlen characters
    const sentences = []
    const nextChars = []
    for (let i = 0; i < text.length; i += SKIP) {
        sentences.push(text.slice(i, i + MAX_LEN))
        nextChars.push(text.slice(i + MAX_LEN, i + MAX_LEN + 1))
        // console.log(text.slice(i, i + MAX_LEN), '->', text.slice(i + MAX_LEN, i + MAX_LEN + 1))
    }
    
    // one-hotify
    const x = tf.zeros([sentences.length, MAX_LEN, chars.length])
    const y = tf.zeros([sentences.length, chars.length])
    sentences.forEach((sentence, i) => {
        sentence.split('').forEach((char, t) => {
            // console.log(char)
            // console.log(i, t, charIndicies[char], char)
            x.buffer().set(1, i, t, charIndicies[char])
        })
        y.buffer().set(1, i, charIndicies[nextChars[i]])
    })
    
    // const tmp = x.dataSync()
    // for (let i = 0; i < 40; i++) {
    //     const t = tmp.slice(i * chars.length, i * chars.length + chars.length + 1)
    //     const tensor = tf.tensor(t).argMax().dataSync()
    //     const num = tensor[0]
    //     console.log(chars[num], num)
    // }
    
    return {
        x, y, charIndicies
    }
} 

async function main() {
    
    const data = await getData(`${__dirname}/data/tinyshakespeare.txt`)
    const model = getModel([MAX_LEN, Object.keys(data.charIndicies).length])
    console.log('fitting model')
    
    for (let i = 0; i < EPOCHS; i++) {
        const history = await model.fit(data.x, data.y) 
        console.log(`loss: ${history.history.loss[0]}, accuracy: ${history.history.acc[0]}`)
    }
    
    data.x.dispose()
    data.y.dispose()
}

main()