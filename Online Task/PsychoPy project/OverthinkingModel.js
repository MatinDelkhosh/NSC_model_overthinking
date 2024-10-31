/************************** 
 * Overthinkingmodel *
 **************************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2024.2.1post4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'OverthinkingModel';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0.5, 0.5, 0.5]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);


flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.INFO);

async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.2.1post4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}

async function experimentInit() {
  // Initialize components for Routine "MazeRoutine"
  MazeRoutineClock = new util.Clock();
  maze_time = new visual.ShapeStim ({
    win: psychoJS.window, name: 'maze_time', 
    vertices: [[-[0.5, 0.5][0]/2.0, -[0.5, 0.5][1]/2.0], [+[0.5, 0.5][0]/2.0, -[0.5, 0.5][1]/2.0], [0, [0.5, 0.5][1]/2.0]],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    fillColor: 'white',
    opacity: 0.0, depth: -1, interpolate: true,
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial of trials) {
      snapshot = trials.getSnapshot();
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(MazeRoutineRoutineBegin(snapshot));
      trialsLoopScheduler.add(MazeRoutineRoutineEachFrame());
      trialsLoopScheduler.add(MazeRoutineRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function MazeRoutineRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'MazeRoutine' ---
    t = 0;
    MazeRoutineClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(20.000000);
    MazeRoutineMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from MazeCode
    import * as random from 'random';
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    class Maze {
        constructor(size = 8) {
            this.size = size;
            this.maze = function () {
        var _pj_a = [], _pj_b = util.range(((2 * size) + 1));
        for (var _pj_c = 0, _pj_d = _pj_b.length; (_pj_c < _pj_d); _pj_c += 1) {
            var _ = _pj_b[_pj_c];
            _pj_a.push(([1] * ((2 * size) + 1)));
        }
        return _pj_a;
    }
    .call(this);
            this.visited = set([]);
            this.goal_location = null;
            this.agent_location = null;
            this.walls = set([]);
        }
        generate_maze() {
            var agent_start, start, wall;
            start = [((Math.random.randint(0, (this.size - 1)) * 2) + 1), ((Math.random.randint(0, (this.size - 1)) * 2) + 1)];
            this.maze[start[1]][start[0]] = 0;
            this.walk_maze(start);
            for (var x, _pj_c = 0, _pj_a = util.range(1, (this.size * 2), 2), _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
                x = _pj_a[_pj_c];
                for (var y, _pj_f = 0, _pj_d = util.range(1, (this.size * 2), 2), _pj_e = _pj_d.length; (_pj_f < _pj_e); _pj_f += 1) {
                    y = _pj_d[_pj_f];
                    this.walls.add([x, y]);
                }
            }
            for (var _, _pj_c = 0, _pj_a = util.range(4), _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
                _ = _pj_a[_pj_c];
                if ((! this.walls)) {
                    break;
                }
                wall = Math.random.choice(list(this.walls));
                this.maze[wall[1]][wall[0]] = 0;
                this.walls.remove(wall);
            }
            this.goal_location = [((Math.random.randint(0, (this.size - 1)) * 2) + 1), ((Math.random.randint(0, (this.size - 1)) * 2) + 1)];
            while (true) {
                agent_start = [((Math.random.randint(0, (this.size - 1)) * 2) + 1), ((Math.random.randint(0, (this.size - 1)) * 2) + 1)];
                if ((agent_start !== this.goal_location)) {
                    this.agent_location = agent_start;
                    break;
                }
            }
            return this.maze;
        }
        walk_maze(s) {
            var neighbors;
            this.visited.add(s);
            neighbors = this.neighbors(s);
            Math.random.shuffle(neighbors);
            for (var n, _pj_c = 0, _pj_a = neighbors, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
                n = _pj_a[_pj_c];
                if ((! _pj.in_es6(n, this.visited))) {
                    this.remove_wall(s, n);
                    this.walk_maze(n);
                }
            }
        }
        neighbors(s) {
            var neighbors, nx, ny, potential_neighbors, x, y;
            [x, y] = s;
            potential_neighbors = [[(x - 2), y], [(x + 2), y], [x, (y - 2)], [x, (y + 2)]];
            neighbors = [];
            for (var n, _pj_c = 0, _pj_a = potential_neighbors, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
                n = _pj_a[_pj_c];
                [nx, ny] = n;
                if ((((0 <= nx) && (nx < ((this.size * 2) + 1))) && ((0 <= ny) && (ny < ((this.size * 2) + 1))))) {
                    neighbors.push([nx, ny]);
                }
            }
            return neighbors;
        }
        remove_wall(s, n) {
            var nx, ny, sx, sy, wall_pos;
            [sx, sy] = s;
            [nx, ny] = n;
            if ((sx === nx)) {
                wall_pos = [sx, (Math.min(sy, ny) + 1)];
                this.maze[wall_pos[1]][wall_pos[0]] = 0;
            } else {
                if ((sy === ny)) {
                    wall_pos = [(Math.min(sx, nx) + 1), sy];
                    this.maze[wall_pos[1]][wall_pos[0]] = 0;
                }
            }
            this.maze[ny][nx] = 0;
            this.walls.discard(wall_pos);
        }
        teleport_agent() {
            var random_location;
            while (true) {
                random_location = [((Math.random.randint(0, (this.size - 1)) * 2) + 1), ((Math.random.randint(0, (this.size - 1)) * 2) + 1)];
                if ((random_location !== this.goal_location)) {
                    return random_location;
                }
            }
        }
    }
    WALL_COLOR = [(- 0.4), (- 0.4), (- 0.4)];
    PATH_COLOR = [0.7, 0.7, 0.7];
    AGENT_COLOR = [(- 1), (- 1), 1];
    GOAL_COLOR = [1, (- 1), (- 1)];
    maz = new Maze({"size": 10});
    maz.generate_maze();
    cell_size = (0.4 / maz.size);
    player_size = 0.9;
    function draw_maze(maze = maz) {
        var agent, ax, ay, color, goal, gx, gy, rect;
        for (var y, _pj_c = 0, _pj_a = util.range(maze.maze.length), _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
            y = _pj_a[_pj_c];
            for (var x, _pj_f = 0, _pj_d = util.range(maze.maze[0].length), _pj_e = _pj_d.length; (_pj_f < _pj_e); _pj_f += 1) {
                x = _pj_d[_pj_f];
                color = ((maze.maze[y][x] === 1) ? WALL_COLOR : PATH_COLOR);
                rect = new visual.Rect(psychoJS.window, {"width": cell_size, "height": cell_size, "fillColor": color});
                rect.pos = [((x * cell_size) - (maze.size * cell_size)), ((y * cell_size) - (maze.size * cell_size))];
                rect.draw();
            }
        }
        [ax, ay] = maze.agent_location;
        agent = new visual.Circle(psychoJS.window, {"radius": ((cell_size * player_size) / 2), "fillColor": AGENT_COLOR});
        agent.pos = [((ax * cell_size) - (maze.size * cell_size)), ((ay * cell_size) - (maze.size * cell_size))];
        agent.draw();
        [gx, gy] = maze.goal_location;
        goal = new visual.Rect(psychoJS.window, {"width": cell_size, "height": cell_size, "fillColor": GOAL_COLOR});
        goal.pos = [((gx * cell_size) - (maze.size * cell_size)), ((gy * cell_size) - (maze.size * cell_size))];
        goal.draw();
    }
    function move_agent(key, pos, maze) {
        var c1, c2, new_pos;
        new_pos = pos.slice(0);
        if ((key === "up")) {
            new_pos[1] += 1;
        } else {
            if ((key === "down")) {
                new_pos[1] -= 1;
            } else {
                if ((key === "left")) {
                    new_pos[0] -= 1;
                } else {
                    if ((key === "right")) {
                        new_pos[0] += 1;
                    }
                }
            }
        }
        c1 = ((0 <= new_pos[1]) && (new_pos[1] < maze.size));
        c2 = ((0 <= new_pos[0]) && (new_pos[0] < maze.size));
        if ((c1 && c2)) {
            if ((maze.maze[new_pos[1]][new_pos[0]] === 0)) {
                return new_pos;
            }
        }
        return pos;
    }
    pressed_keys = [];
    frame = 0;
    
    psychoJS.experiment.addData('MazeRoutine.started', globalClock.getTime());
    MazeRoutineMaxDuration = null
    // keep track of which components have finished
    MazeRoutineComponents = [];
    MazeRoutineComponents.push(maze_time);
    
    for (const thisComponent of MazeRoutineComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function MazeRoutineRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'MazeRoutine' ---
    // get current time
    t = MazeRoutineClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // Run 'Each Frame' code from MazeCode
    keys = psychoJS.eventManager.getKeys();
    frame += 1;
    for (var key, _pj_c = 0, _pj_a = keys, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
        key = _pj_a[_pj_c];
        console.log(key, frame);
        maz.agent_location = move_agent(key, maz.agent_location, maz);
    }
    draw_maze(maz);
    
    
    // *maze_time* updates
    if (t >= 0.0 && maze_time.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      maze_time.tStart = t;  // (not accounting for frame time here)
      maze_time.frameNStart = frameN;  // exact frame index
      
      maze_time.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 20 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (maze_time.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      maze_time.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of MazeRoutineComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function MazeRoutineRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'MazeRoutine' ---
    for (const thisComponent of MazeRoutineComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('MazeRoutine.stopped', globalClock.getTime());
    if (MazeRoutineMaxDurationReached) {
        routineTimer.add(MazeRoutineMaxDuration);
    } else {
        routineTimer.add(-20.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}

async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
