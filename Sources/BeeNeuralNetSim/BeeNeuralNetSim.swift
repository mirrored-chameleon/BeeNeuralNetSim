// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import SwiftNN

// Helper variabels to store info about the simulation, like what emojis correspond to what, the size of the grid and such
let objectValuesForward: [String: Double] = ["🐝": 1, "🏮": 4, "🌻": 5, "P": 3, "🌿": 0]
let objectValuesBackward: [Double: String] = [1: "🐝", 4: "🏮", 5: "🌻", 3: "P", 0: "🌿"]

let gridValueX = 20
let gridValueY = 20

let learningRate = 0.1
@MainActor var explorationRate = 0.15
let maxEpisodes = 80
let maxStepsPerEpisode = 100
let simulationDelay = 0.2

// Renders frames to the terminal so we can see the simulation happening.
func renderFrame(_ frame: String, foundFlowerAndHive: Bool) {
    print("\u{001B}[H\(frame)", terminator: "")
    if foundFlowerAndHive {
        print("Found flower and hive!!")
    }
    fflush(stdout)
}

// Calculates the distance from one point to another on the grid.
func distance(from bee: (x: Int, y: Int), to flower: (x: Int, y: Int)) -> Double {
    return Double(abs(bee.x - flower.x) + abs(bee.y - flower.y))
}

// An encoder to help the model, and this one takes in more info than the one in the talent, which we are not using for this test.
func encodeGrid(_ input: String, hasFlower: Bool, beeX: Int, beeY: Int, targetX: Int, targetY: Int, lastMove: Int)
    -> Matrix<Double>
{

    var values = input.map { char in
        objectValuesForward[char.description] ?? 0.0
    }

    values.append(hasFlower ? 1.0 : 0.0)

    let dx = Double(targetX - beeX) / Double(gridValueX)
    let dy = Double(targetY - beeY) / Double(gridValueY)

    values.append(dx)
    values.append(dy)

    values.append(Double(lastMove) / 3.0)

    return Matrix<Double>(rows: 1, columns: values.count, grid: values)
}

// Bee talent for the Buzzy Bee himself
struct BeeTalent: Talent {

    typealias Input = String

    typealias Output = Int

    var tokens: [Int] = [0, 1, 2, 3]

    func encode(_ input: String) -> SwiftNN.Matrix<Double> {
        var output = Matrix<Double>(rows: gridValueX, columns: gridValueY, grid: [])

        for elem in input {
            output.grid.append(objectValuesForward[elem.description] ?? 0.0)
        }

        return output
    }

    func decode(_ output: SwiftNN.Matrix<Double>) -> Int {
        let values = flatten(output)

        let index = argmax(values)
        return tokens[index]
    }
}

// Buzzy Bee himself, look at him in all his glory!
// All jokes aside, this is the heart of the test, the network that predicts, trains and finds pollen for us
struct BeeNetwork: Network {
    var talent: any SwiftNN.Talent

    var weights: [[SwiftNN.Matrix<Double>]]

    var bias: [SwiftNN.Matrix<Double>]

    mutating func train(input: SwiftNN.Matrix<Double>, target: SwiftNN.Matrix<Double>) {

    }

    mutating func predict(input: SwiftNN.Matrix<Double>) -> SwiftNN.Matrix<Double> {
        var currentOutput = input

        for currentLayerIndex in 0..<weights.count {

            let currentLayerWeights = weights[currentLayerIndex]
            let currentLayerBias = bias[currentLayerIndex]

            var nextLayerOutputMatrix = Matrix<Double>(
                rows: 1,
                columns: currentLayerWeights.count,
                grid: Array(repeating: 0.0, count: currentLayerWeights.count)
            )

            for neuronIndexInLayer in 0..<currentLayerWeights.count {

                let neuronWeightMatrix = currentLayerWeights[neuronIndexInLayer]

                var weightedSum: Double = 0.0

                for inputIndex in 0..<neuronWeightMatrix.columns {

                    let inputValue = currentOutput.grid[inputIndex]
                    let weightValue = neuronWeightMatrix.grid[inputIndex]

                    weightedSum += inputValue * weightValue
                }

                let biasValue = currentLayerBias.grid[neuronIndexInLayer]

                nextLayerOutputMatrix.grid[neuronIndexInLayer] = weightedSum + biasValue
            }

            currentOutput = nextLayerOutputMatrix

            if currentLayerIndex < weights.count - 1 {
                currentOutput = reluMatrix(currentOutput)
            }
        }

        return Matrix<Double>(
            rows: currentOutput.rows, columns: currentOutput.columns,
            grid: softmax(currentOutput.grid))
    }

    func hiddenStateBeforeOutput(input: SwiftNN.Matrix<Double>) -> SwiftNN.Matrix<Double> {
        var currentOutput = input

        if weights.count <= 1 {
            return currentOutput
        }

        for currentLayerIndex in 0..<(weights.count - 1) {
            let currentLayerWeights = weights[currentLayerIndex]
            let currentLayerBias = bias[currentLayerIndex]

            var nextLayerOutputMatrix = Matrix<Double>(
                rows: 1,
                columns: currentLayerWeights.count,
                grid: Array(repeating: 0.0, count: currentLayerWeights.count)
            )

            for neuronIndexInLayer in 0..<currentLayerWeights.count {
                let neuronWeightMatrix = currentLayerWeights[neuronIndexInLayer]
                var weightedSum: Double = 0.0

                for inputIndex in 0..<neuronWeightMatrix.columns {
                    let inputValue = currentOutput.grid[inputIndex]
                    let weightValue = neuronWeightMatrix.grid[inputIndex]

                    weightedSum += inputValue * weightValue
                }

                let biasValue = currentLayerBias.grid[neuronIndexInLayer]
                nextLayerOutputMatrix.grid[neuronIndexInLayer] = weightedSum + biasValue
            }

            currentOutput = reluMatrix(nextLayerOutputMatrix)
        }

        return currentOutput
    }

    mutating func trainRL(points: Double, move: Int, input: Matrix<Double>) {

        let reward = tanh(points / 10.0) * 5.0
        guard let outputLayerIndex = weights.indices.last else {
            return
        }

        if move >= 0 && move < weights[outputLayerIndex].count {
            let hiddenState = hiddenStateBeforeOutput(input: input)
            var moveWeights = weights[outputLayerIndex][move]

            for i in 0..<moveWeights.grid.count {
                let activation = hiddenState.grid[i]
                let signal = activation / (1.0 + abs(activation))
                let lr = learningRate * 0.5
                moveWeights.grid[i] += lr * reward * signal
            }

            weights[outputLayerIndex][move] = moveWeights
            bias[outputLayerIndex].grid[move] += learningRate * reward
        }
    }

    public init(layers: [Int], talent: any Talent) {

        self.talent = talent
        self.weights = []
        self.bias = []

        for layerIndex in 0..<(layers.count - 1) {

            let inputSize = layers[layerIndex]
            let outputSize = layers[layerIndex + 1]

            let initializationScale =
                sqrt(2.0 / Double(inputSize))

            var layerWeights: [Matrix<Double>] = []

            for _ in 0..<outputSize {

                let weightMatrix = Matrix(
                    rows: inputSize,
                    columns: 1,
                    grid: (0..<inputSize).map { _ in
                        Double.random(in: -1...1) * initializationScale
                    }
                )

                layerWeights.append(weightMatrix)
            }

            weights.append(layerWeights)

            let layerBias = Matrix(
                rows: outputSize,
                columns: 1,
                grid: Array(repeating: 0.0, count: outputSize)
            )

            bias.append(layerBias)
        }
    }

}

// The test to test the model, generating the grid, and you can pretty much change most things about it
@main
struct BeeNeuralNetSim {

    static func main() {

        var statusMessage = ""

        var frames: [String] = []

        let gridValueX: Int = 20
        let gridValueY = 20

        let beeTalent = BeeTalent()

        var foundFlowerAndHive = false

        var beeModel = BeeNetwork(
            layers: [gridValueX * gridValueY + 3, 64, 64, 64, 4], talent: beeTalent)

        print("\u{001B}[2J\u{001B}[?25l", terminator: "")
        defer {
            print("\u{001B}[?25h")
        }

        var lastMove = 0

        for episode in 1...maxEpisodes {

            
            let decay = 0.92
            explorationRate = max(0.005, 0.15 * pow(decay, Double(episode)))


            var beeX = 5
            var beeY = 5
            var visitedPositions = Set<String>()
            var flowers: [(isHidden: Bool, coordinates: (x: Int, y: Int))] = []

            for _ in 0..<3 {
                flowers.append(
                    (
                        false,
                        (
                            Int.random(in: 0..<gridValueX),
                            Int.random(in: 0..<gridValueY)
                        )
                    ))
            }
            var foundFlower = false

            for step in 1...maxStepsPerEpisode {

                var grid = ""
                var frame = "Episode \(episode), step \(step)\u{001B}[K\n"

                let hiveX = 10
                let hiveY = 10

                for y in 0..<gridValueY {
                    var row = ""

                    for x in 0..<gridValueX {
                        if x == beeX && y == beeY {
                            row += "🐝"
                        } else if x == hiveX && y == hiveY {
                            row += "🏮"
                        } else {
                            var hasVisibleFlower = false

                            for flower in flowers {
                                if flower.coordinates.x == x && flower.coordinates.y == y
                                    && flower.isHidden == false
                                {
                                    hasVisibleFlower = true
                                    break
                                }
                            }

                            row += hasVisibleFlower ? "🌻" : "🌿"
                        }

                    }

                    grid += row

                    frame += "\(row)\u{001B}[K\n"

                }

                frame += "\n\(statusMessage)\u{001B}[K\n"
                frames.append(frame)

                guard let targetFlower = flowers.first(where: { $0.isHidden == false }) else {
                    break
                }

                let flowerDifference = distance(
                    from: (x: beeX, y: beeY), to: targetFlower.coordinates)
                let hiveDifference = distance(from: (x: beeX, y: beeY), to: (x: hiveX, y: hiveY))
                let targetX = foundFlower ? hiveX : targetFlower.coordinates.x
                let targetY = foundFlower ? hiveY : targetFlower.coordinates.y

                let input = encodeGrid(
                    grid,
                    hasFlower: foundFlower,
                    beeX: beeX,
                    beeY: beeY,
                    targetX: targetX,
                    targetY: targetY,
                    lastMove: lastMove
                )

                let prediction = beeModel.predict(input: input)
                let move =
                    Double.random(in: 0.0...1.0) < explorationRate
                    ? beeTalent.tokens.randomElement()!
                    : beeTalent.decode(prediction)

                switch move {
                case 0: beeX += 1
                case 1: beeX -= 1
                case 2: beeY += 1
                case 3: beeY -= 1
                default: break
                }

                beeX = max(0, min(gridValueX - 1, beeX))
                beeY = max(0, min(gridValueY - 1, beeY))

                var reward = -0.1  // small step penalty to encourage efficiency

                let positionKey = "\(beeX),\(beeY)"

                if visitedPositions.contains(positionKey) {
                   reward -= 2.0
                }

                let newFlowerDifference = distance(
                    from: (x: beeX, y: beeY), to: targetFlower.coordinates)
                let newHiveDifference = distance(from: (x: beeX, y: beeY), to: (x: hiveX, y: hiveY))

                for indexFlower in 0..<flowers.count {
                    var flower = flowers[indexFlower]
                    if beeX == flower.coordinates.x && beeY == flower.coordinates.y
                        && foundFlower == false
                    {
                        flower.isHidden = true
                        flowers[indexFlower] = flower
                        foundFlower = true
                        print(
                            "Found flower on episode \(episode), step \(step)"
                        )
                    }
                }

                if !foundFlower {
                    // Phase 1: ONLY learn flower navigation
                    reward += newFlowerDifference < flowerDifference ? 2.0 : -2.0

                    if beeX == targetFlower.coordinates.x && beeY == targetFlower.coordinates.y {
                        reward += 5.0
                        foundFlower = true
                    }

                } else {
                    // Phase 2: ONLY learn hive navigation
                    reward += newHiveDifference < hiveDifference ? 1.0 : -1.0

                    if beeX == hiveX && beeY == hiveY {
                        reward += 20.0
                        foundFlowerAndHive = true
                        break
                    }
                }

                beeModel.trainRL(points: reward, move: move, input: input)

                visitedPositions.insert(positionKey)

                lastMove = move

                if foundFlowerAndHive {
                    break
                }
            }
            if foundFlowerAndHive {
                break
            } else {
                frames = []
            }
        }

        if foundFlowerAndHive == false {
            print(
                "Did not find flower and hive after \(maxEpisodes) episodes"
            )
        } else {
            print("Found hive and flower!!")
        }

        if foundFlowerAndHive {
            for frame in frames {
                renderFrame(frame, foundFlowerAndHive: foundFlowerAndHive)
                Thread.sleep(forTimeInterval: simulationDelay)
            }
        }
    }
}
