const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

// Initialize the Express app and Socket.io
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: "*" } // Allow phones to connect from any browser
});

// Memory storage for incoming client weights for the current round
let clientWeightsRound = [];
const MIN_CLIENTS_FOR_AGGREGATION = 5; // Wait for at least 5 phones before averaging

// Serve static files (this will host your Dashboard and Mobile App)
app.use(express.static('public'));

io.on('connection', (socket) => {
    console.log(`New phone connected: ${socket.id}`);

    // 1. Listen for local model updates from audience phones
    socket.on('submit_local_weights', (localWeights) => {
        console.log(`Received weights from ${socket.id}`);
        clientWeightsRound.push(localWeights);

        // 2. Check if we have enough participants to run FedAvg
        if (clientWeightsRound.length >= MIN_CLIENTS_FOR_AGGREGATION) {
            console.log(`Threshold reached. Running FedAvg on ${clientWeightsRound.length} models...`);
            
            const globalWeights = performFedAvg(clientWeightsRound);
            
            // 3. Broadcast the new global model back to everyone
            io.emit('global_model_update', globalWeights);
            console.log('Global model broadcasted to audience.');
            
            // Clear the array for the next round
            clientWeightsRound = []; 
        }
    });

    socket.on('disconnect', () => {
        console.log(`Phone disconnected: ${socket.id}`);
    });
});

/**
 * Performs Federated Averaging on an array of weight sets.
 * Assumes all clients send weights in the exact same array structure/shape.
 */
function performFedAvg(allClientWeights) {
    const numClients = allClientWeights.length;
    const numLayers = allClientWeights[0].length;
    let newGlobalWeights = [];

    // Iterate over each layer of the neural network
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        let layerSum = null;

        // Iterate over each client's submitted weights
        for (let clientIdx = 0; clientIdx < numClients; clientIdx++) {
            const clientLayer = allClientWeights[clientIdx][layerIdx];
            
            if (layerSum === null) {
                // Initialize the sum array with the first client's dimensions
                layerSum = new Array(clientLayer.length).fill(0);
            }

            // Add the weights element-wise
            for (let i = 0; i < clientLayer.length; i++) {
                layerSum[i] += clientLayer[i];
            }
        }

        // Divide by the number of clients to get the average
        const layerAverage = layerSum.map(val => val / numClients);
        newGlobalWeights.push(layerAverage);
    }

    return newGlobalWeights;
}

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Federated Server running on port ${PORT}`);
});