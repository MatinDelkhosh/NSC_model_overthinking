const canvas = document.getElementById('mazeCanvas');
const ctx = canvas.getContext('2d');

// Maze settings
const rows = 10;  // Change this for different maze sizes
const cols = 10;
const cellSize = canvas.width / cols;

// Player position
let player = { x: 0, y: 0 };
let goal = { x: cols - 1, y: rows - 1 };

// Directions for movement
const directions = [
  { dx: 1, dy: 0 },  // Right
  { dx: -1, dy: 0 }, // Left
  { dx: 0, dy: 1 },  // Down
  { dx: 0, dy: -1 }  // Up
];

// Maze data structures
let maze = [];
let visited = [];

// Initialize maze and visited grid
for (let row = 0; row < rows; row++) {
  maze[row] = [];
  visited[row] = [];
  for (let col = 0; col < cols; col++) {
    maze[row][col] = { top: true, right: true, bottom: true, left: true };
    visited[row][col] = false;
  }
}

// Utility function to shuffle array (used for randomizing directions)
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// Recursive function to generate the maze using DFS
function generateMaze(x, y) {
  visited[x][y] = true;
  shuffle(directions);  // Randomize the directions to make it more interesting

  for (let dir of directions) {
    const newX = x + dir.dx;
    const newY = y + dir.dy;

    // Check if the new cell is within bounds and not visited
    if (newX >= 0 && newX < cols && newY >= 0 && newY < rows && !visited[newX][newY]) {
      // Remove walls between current cell and new cell
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

      // Mark the new cell as part of the current path
      maze[newX][newY].partOfPath = true;

      // Recursively generate the maze from the new cell
      generateMaze(newX, newY);
    }
  }
}

// Draw the maze on the canvas
function drawMaze() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);  // Clear the canvas
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

// Move the player
function movePlayer(dx, dy) {
  const newX = player.x + dx;
  const newY = player.y + dy;

  // Check if the new cell is within bounds and there is no wall
  if (newX >= 0 && newX < cols && newY >= 0 && newY < rows) {
    if (dx === 1 && !maze[player.y][player.x].right) player.x = newX;
    if (dx === -1 && !maze[player.y][player.x].left) player.x = newX;
    if (dy === 1 && !maze[player.y][player.x].bottom) player.y = newY;
    if (dy === -1 && !maze[player.y][player.x].top) player.y = newY;
  }

  // Check if player has reached the goal
  if (player.x === goal.x && player.y === goal.y) {
    document.getElementById('status').innerText = 'You Win!';
  }

  drawMaze();
}

// Listen for keyboard input to move the player
document.addEventListener('keydown', function (e) {
  if (e.key === 'ArrowUp') movePlayer(0, -1);
  if (e.key === 'ArrowDown') movePlayer(0, 1);
  if (e.key === 'ArrowLeft') movePlayer(-1, 0);
  if (e.key === 'ArrowRight') movePlayer(1, 0);
});

// Start the game by generating the maze and drawing it
generateMaze(0, 0);
drawMaze();