const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { spawn } = require('child_process');
const cors = require('cors');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "http://localhost:3000",  // Adjust this to your client's origin
        methods: ["GET", "POST"]
    }
});

// Start the Python script for processing
const pythonProcess = spawn('python', ['app.py']);

// Middleware to handle CORS
app.use(cors());

// Socket.IO connection
io.on('connection', (socket) => {
    console.log('A user connected');

    // Handle incoming video frames
    socket.on('video_frame', (data) => {
        // Forward the data to the Python process
        pythonProcess.stdin.write(data + '\n');
    });

    // Handle processed frames (bounding boxes) from the Python process
    pythonProcess.stdout.on('data', (data) => {
        const boxes = JSON.parse(data.toString().trim());
        socket.emit('processed_frame', boxes); // Send boxes to the frontend
    });

    // Handle errors
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    socket.on('disconnect', () => {
        console.log('A user disconnected');
    });
});

// Start the server
server.listen(5000, () => {
    console.log('Server is running on port 5000');
});
