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
        .package(url: "https://github.com/mirrored-chameleon/SwiftNN", branch: "main")
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