const canvas = document.getElementById('mazeCanvas');
const ctx = canvas.getContext('2d');
const rows = 15;
const cols = 15;
const cellSize = canvas.width / cols;

// Player position
let player = { x: 0, y: 0 };
let goal = { x: cols - 1, y: rows - 1 };

// Timer
let timeLeft = 20;
const timerElement = document.getElementById("timer");
const gameOverElement = document.getElementById("gameOver");

// Initialize maze and visited grid
let maze = [];
let visited = [];

// Directions for movement
const directions = [
  { dx: 1, dy: 0 },  // Right
  { dx: -1, dy: 0 }, // Left
  { dx: 0, dy: 1 },  // Down
  { dx: 0, dy: -1 }  // Up
];

// Maze generation using Depth-First Search (DFS)
function generateMaze() {
  // Initialize the maze and visited array
  for (let row = 0; row < rows; row++) {
    maze[row] = [];
    visited[row] = [];
    for (let col = 0; col < cols; col++) {
      maze[row][col] = { top: true, right: true, bottom: true, left: true };
      visited[row][col] = false;
    }
  }

  // Recursive DFS to carve the maze
  function dfs(x, y) {
    visited[x][y] = true;
    shuffle(directions);  // Randomize directions

    for (let dir of directions) {
      const newX = x + dir.dx;
      const newY = y + dir.dy;

      // Check if the new cell is within bounds and unvisited
      if (newX >= 0 && newX < rows && newY >= 0 && newY < cols && !visited[newX][newY]) {
        // Carve the walls
        if (dir.dx === 1) { // Moving right
          maze[x][y].right = false;
          maze[newX][newY].left = false;
        } else if (dir.dx === -1) { // Moving left
          maze[x][y].left = false;
          maze[newX][newY].right = false;
        } else if (dir.dy === 1) { // Moving down
          maze[x][y].bottom = false;
          maze[newX][newY].top = false;
        } else if (dir.dy === -1) { // Moving up
          maze[x][y].top = false;
          maze[newX][newY].bottom = false;
        }

        dfs(newX, newY); // Recursively visit the next cell
      }
    }
  }

  dfs(0, 0); // Start maze generation from the top-left corner
}

// Utility function to shuffle an array (Fisher-Yates Shuffle)
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// Function to draw the maze
function drawMaze() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const x = col * cellSize;
      const y = row * cellSize;

      // Draw walls
      if (maze[row][col].top) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + cellSize, y);
        ctx.stroke();
      }
      if (maze[row][col].right) {
        ctx.beginPath();
        ctx.moveTo(x + cellSize, y);
        ctx.lineTo(x + cellSize, y + cellSize);
        ctx.stroke();
      }
      if (maze[row][col].bottom) {
        ctx.beginPath();
        ctx.moveTo(x + cellSize, y + cellSize);
        ctx.lineTo(x, y + cellSize);
        ctx.stroke();
      }
      if (maze[row][col].left) {
        ctx.beginPath();
        ctx.moveTo(x, y + cellSize);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
  }

  // Draw the player
  ctx.fillStyle = 'blue';
  ctx.fillRect(player.x * cellSize + cellSize / 4, player.y * cellSize + cellSize / 4, cellSize / 2, cellSize / 2);

  // Draw the goal
  ctx.fillStyle = 'green';
  ctx.fillRect(goal.x * cellSize + cellSize / 4, goal.y * cellSize + cellSize / 4, cellSize / 2, cellSize / 2);
}

// Function to handle player movement
function movePlayer(dx, dy) {
  const newX = player.x + dx;
  const newY = player.y + dy;

  if (newX >= 0 && newX < cols && newY >= 0 && newY < rows) {
    // Check if there is a wall blocking the move
    if (dx === 1 && !maze[player.y][player.x].right) player.x = newX;
    if (dx === -1 && !maze[player.y][player.x].left) player.x = newX;
    if (dy === 1 && !maze[player.y][player.x].bottom) player.y = newY;
    if (dy === -1 && !maze[player.y][player.x].top) player.y = newY;
  }

  // Check if player reached the goal
  if (player.x === goal.x && player.y === goal.y) {
    endGame(true);
  }
}

// Function to end the game
function endGame(win) {
  clearInterval(timerInterval);
  gameOverElement.style.display = 'block';
  gameOverElement.innerText = win ? 'You Win!' : 'Time\'s Up!';
}

// Countdown timer function
function updateTimer() {
  timeLeft -= 1;
  timerElement.innerText = `Time Left: ${timeLeft} seconds`;

  if (timeLeft <= 0) {
    endGame(false);
  }
}

// Keydown event listener for player movement
document.addEventListener('keydown', function (e) {
  if (e.key === 'ArrowUp') movePlayer(0, -1);
  if (e.key === 'ArrowDown') movePlayer(0, 1);
  if (e.key === 'ArrowLeft') movePlayer(-1, 0);
  if (e.key === 'ArrowRight') movePlayer(1, 0);

  drawMaze();
});

// Start the game
function startGame() {
  generateMaze();
  drawMaze();
  timerInterval = setInterval(updateTimer, 1000);
}

let timerInterval;
startGame();
