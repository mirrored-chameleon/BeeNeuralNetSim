// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "BeeNeuralNetSim",
    products: [
        .executable(
            name: "BeeNeuralNetSim",
            targets: ["BeeNeuralNetSim"]
        )
    ],
    dependencies: [
        .package(path: "../../Documents/GitHub/SwiftNN/SwiftNN")
    ],
    targets: [
        .executableTarget(
            name: "BeeNeuralNetSim",
            dependencies: [
                "SwiftNN"
            ]
        ),
    ]
)