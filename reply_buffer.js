/**
 * https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
 * 
 * Returns a random integer between min (inclusive) and max (inclusive).
 * The value is no lower than min (or the next integer greater than min
 * if min isn't an integer) and no greater than max (or the next integer
 * lower than max if max isn't an integer).
 * Using Math.round() will give you a non-uniform distribution!
 */
 const getRandomInt = (min, max)  => {
    min = Math.ceil(min)
    max = Math.floor(max)
    return Math.floor(Math.random() * (max - min + 1)) + min
}

/**
 * Reply Buffer.
 */
class ReplyBuffer {
    constructor({
        size = 500, // number of stored transitions
    }) {
        this._size = size
        
        this._buffer = new Array(size).fill()
        this._pool = []

        this.length = 0
    }

    add(transition) {
        let {id, priority = 1} = transition

        if (priority < 1) 
            throw new Error('Priority less than 1')

        const index = id % this._size

        if (this._buffer[index])
            this._pool = this._pool.filter(i => i !== index)
        else
            this.length++

        while (priority--) 
            this._pool.push(index)

        this._buffer[index] = transition
    }

    sample(n) {
        if (this.length < n) 
            return []

        const 
            sampleIndices = new Set(),
            samples = []

        while (n--)
            while (sampleIndices.size < this.length) {
                const randomIndex = this._pool[getRandomInt(0, this._pool.length - 1)]
                if (sampleIndices.has(randomIndex))
                    continue

                sampleIndices.add(randomIndex)

                const { id, state, action, reward } = this._buffer[randomIndex]
                const nextId = id + 1
                const next = this._buffer[nextId % this._size]

                if (next && next.id === nextId) {
                    samples.push({ id, nextId: next.id, state, action, reward, nextState: next.state})
                    break
                }
            }

        return samples.length == n ? samples : []
    }
}

/** TESTS */
(() => {
    const rb = new ReplyBuffer({size: 5})
    rb.add({id: 0})
    rb.add({id: 1})
    rb.add({id: 2, priority: 3})
    
    console.assert(rb.length === 3)
    console.assert(rb._pool.length === 5)
    console.assert(rb._buffer[0].id === 0)
    
    rb.add({id: 2})
    rb.add({id: 4, state: 4, priority: 2})
    
    console.assert(rb.length === 4)
    console.assert(rb._pool.length === 5)
    console.assert(JSON.stringify(rb._pool) === '[0,1,2,4,4]')
    
    rb.add({id: 5, priority: 2})
    
    console.assert(rb.length === 4)
    console.assert(rb._pool.length === 6)
    console.assert(rb._buffer.length === 5)
    console.assert(rb._buffer[0].id === 5)
    console.assert(JSON.stringify(rb._pool) === '[1,2,4,4,0,0]')

    console.assert(rb.sample(999).length === 0, 'Too many samples')
    
    console.log(rb._buffer)
    console.assert(rb.sample(2).length === 2, 'Only two samples possible')

    rb.add({id: 506, priority: 3})

    const samples = rb.sample(1)
    console.assert(samples.length === 1, 'Only one suitable sample with valid next state')
    console.assert(samples[0].state === 4, 'Sample with id:4')

    console.assert(rb.sample(2).length === 0, 
        'Can not sample 2 transitions since next state is available only for one state')
})()
