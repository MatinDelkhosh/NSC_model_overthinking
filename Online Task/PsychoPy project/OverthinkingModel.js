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
flowScheduler.add(welcomeRoutineBegin());
flowScheduler.add(welcomeRoutineEachFrame());
flowScheduler.add(welcomeRoutineEnd());
const questionnaires_trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(questionnaires_trialsLoopBegin(questionnaires_trialsLoopScheduler));
flowScheduler.add(questionnaires_trialsLoopScheduler);
flowScheduler.add(questionnaires_trialsLoopEnd);


flowScheduler.add(wait_for_mazeRoutineBegin());
flowScheduler.add(wait_for_mazeRoutineEachFrame());
flowScheduler.add(wait_for_mazeRoutineEnd());
flowScheduler.add(wait_a_secRoutineBegin());
flowScheduler.add(wait_a_secRoutineEachFrame());
flowScheduler.add(wait_a_secRoutineEnd());
const maze_trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(maze_trialsLoopBegin(maze_trialsLoopScheduler));
flowScheduler.add(maze_trialsLoopScheduler);
flowScheduler.add(maze_trialsLoopEnd);


flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'questioner.xlsx', 'path': 'questioner.xlsx'},
    {'name': 'que/q1.png', 'path': 'que/q1.png'},
    {'name': 'que/q2.png', 'path': 'que/q2.png'},
    {'name': 'que/q3.png', 'path': 'que/q3.png'},
    {'name': 'que/q4.png', 'path': 'que/q4.png'},
    {'name': 'que/q5.png', 'path': 'que/q5.png'},
    {'name': 'que/q6.png', 'path': 'que/q6.png'},
    {'name': 'que/q7.png', 'path': 'que/q7.png'},
    {'name': 'que/q8.png', 'path': 'que/q8.png'},
    {'name': 'que/q9.png', 'path': 'que/q9.png'},
    {'name': 'que/q10.png', 'path': 'que/q10.png'},
    {'name': 'que/q11.png', 'path': 'que/q11.png'},
    {'name': 'que/q12.png', 'path': 'que/q12.png'},
    {'name': 'que/q13.png', 'path': 'que/q13.png'},
    {'name': 'que/q14.png', 'path': 'que/q14.png'},
    {'name': 'que/q15.png', 'path': 'que/q15.png'},
    {'name': 'default.png', 'path': 'https://pavlovia.org/assets/default/default.png'},
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
  // Initialize components for Routine "welcome"
  welcomeClock = new util.Clock();
  Welcome = new visual.TextStim({
    win: psychoJS.window,
    name: 'Welcome',
    text: 'Thank you for participating\nin our experiment',
    font: 'Arial Bold',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('Black'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "questionnaire"
  questionnaireClock = new util.Clock();
  questionsimages = new visual.ImageStim({
    win : psychoJS.window,
    name : 'questionsimages', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0.3], 
    draggable: false,
    size : [1.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  slider = new visual.Slider({
    win: psychoJS.window, name: 'slider',
    startValue: undefined,
    size: [1.0, 0.1], pos: [0, (- 0.2)], ori: 0.0, units: psychoJS.window.units,
    labels: [0, 1, 2, 3, 4], fontSize: 0.05, ticks: [],
    granularity: 1, style: ["RADIO"],
    color: new util.Color('LightGray'), markerColor: new util.Color('Red'), lineColor: new util.Color('White'), 
    opacity: undefined, fontFamily: 'Open Sans', bold: true, italic: false, depth: -1, 
    flip: false,
  });
  
  next = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  mouse = new core.Mouse({
    win: psychoJS.window,
  });
  mouse.mouseClock = new util.Clock();
  // Initialize components for Routine "wait_for_maze"
  wait_for_mazeClock = new util.Clock();
  maze_explanation = new visual.TextStim({
    win: psychoJS.window,
    name: 'maze_explanation',
    text: 'In the next part you are expected to solve 10 mazes in the given time. \nyou can move your charachter with arrow keys.\n\npress SPACE when ready',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "wait_a_sec"
  wait_a_secClock = new util.Clock();
  loading = new visual.TextStim({
    win: psychoJS.window,
    name: 'loading',
    text: 'Loading',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "MazeRoutine"
  MazeRoutineClock = new util.Clock();
  // Run 'Begin Experiment' code from Maze_Code
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
          this.goal_reached = false;
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
          this.goal_location = [(((this.size - 1) * 2) + 1), (((this.size - 1) * 2) + 1)];
          while (true) {
              agent_start = [1, 1];
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
      move_agent(key, pos) {
          var c1, c2, new_pos;
          new_pos = pos.slice(0);
          if (((key === "up") || (key === "w"))) {
              new_pos[1] += 1;
          } else {
              if (((key === "down") || (key === "s"))) {
                  new_pos[1] -= 1;
              } else {
                  if (((key === "left") || (key === "a"))) {
                      new_pos[0] -= 1;
                  } else {
                      if (((key === "right") || (key === "d"))) {
                          new_pos[0] += 1;
                      }
                  }
              }
          }
          c1 = ((0 <= new_pos[1]) && (new_pos[1] < (this.size * 2)));
          c2 = ((0 <= new_pos[0]) && (new_pos[0] < (this.size * 2)));
          if ((c1 && c2)) {
              if ((this.maze[new_pos[1]][new_pos[0]] === 0)) {
                  this.agent_location = new_pos;
                  if ((this.agent_location === this.goal_location)) {
                      this.goal_reached = true;
                  }
              }
          } else {
              this.agent_location = pos;
          }
      }
      draw_maze() {
          var color, rect, rects;
          rects = [];
          for (var y, _pj_c = 0, _pj_a = util.range(this.maze.length), _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
              y = _pj_a[_pj_c];
              for (var x, _pj_f = 0, _pj_d = util.range(this.maze[0].length), _pj_e = _pj_d.length; (_pj_f < _pj_e); _pj_f += 1) {
                  x = _pj_d[_pj_f];
                  color = ((this.maze[y][x] === 1) ? WALL_COLOR : PATH_COLOR);
                  rect = new visual.Rect(psychoJS.window, {"width": cell_size, "height": cell_size, "fillColor": color, "interpolate": true});
                  rect.pos = [((x * cell_size) - (this.size * cell_size)), ((y * cell_size) - (this.size * cell_size))];
                  rects.push(rect);
              }
          }
          return rects;
      }
      draw_agent() {
          var agent, ax, ay, goal, gx, gy;
          [ax, ay] = this.agent_location;
          agent = new visual.Circle(psychoJS.window, {"radius": ((cell_size * player_size) / 2), "fillColor": AGENT_COLOR});
          agent.pos = [((ax * cell_size) - (this.size * cell_size)), ((ay * cell_size) - (this.size * cell_size))];
          agent.draw();
          [gx, gy] = this.goal_location;
          goal = new visual.Rect(psychoJS.window, {"width": cell_size, "height": cell_size, "fillColor": GOAL_COLOR});
          goal.pos = [((gx * cell_size) - (this.size * cell_size)), ((gy * cell_size) - (this.size * cell_size))];
          goal.draw();
      }
  }
  
  Maze_timer = new visual.ShapeStim ({
    win: psychoJS.window, name: 'Maze_timer', 
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

function welcomeRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'welcome' ---
    t = 0;
    welcomeClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(1.500000);
    welcomeMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('welcome.started', globalClock.getTime());
    welcomeMaxDuration = null
    // keep track of which components have finished
    welcomeComponents = [];
    welcomeComponents.push(Welcome);
    
    for (const thisComponent of welcomeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function welcomeRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'welcome' ---
    // get current time
    t = welcomeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Welcome* updates
    if (t >= 0.0 && Welcome.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Welcome.tStart = t;  // (not accounting for frame time here)
      Welcome.frameNStart = frameN;  // exact frame index
      
      Welcome.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (Welcome.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      Welcome.setAutoDraw(false);
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
    for (const thisComponent of welcomeComponents)
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

function welcomeRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'welcome' ---
    for (const thisComponent of welcomeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('welcome.stopped', globalClock.getTime());
    if (welcomeMaxDurationReached) {
        routineTimer.add(welcomeMaxDuration);
    } else {
        routineTimer.add(-1.500000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function questionnaires_trialsLoopBegin(questionnaires_trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    questionnaires_trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'questioner.xlsx',
      seed: undefined, name: 'questionnaires_trials'
    });
    psychoJS.experiment.addLoop(questionnaires_trials); // add the loop to the experiment
    currentLoop = questionnaires_trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisQuestionnaires_trial of questionnaires_trials) {
      snapshot = questionnaires_trials.getSnapshot();
      questionnaires_trialsLoopScheduler.add(importConditions(snapshot));
      questionnaires_trialsLoopScheduler.add(questionnaireRoutineBegin(snapshot));
      questionnaires_trialsLoopScheduler.add(questionnaireRoutineEachFrame());
      questionnaires_trialsLoopScheduler.add(questionnaireRoutineEnd(snapshot));
      questionnaires_trialsLoopScheduler.add(questionnaires_trialsLoopEndIteration(questionnaires_trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function questionnaires_trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(questionnaires_trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function questionnaires_trialsLoopEndIteration(scheduler, snapshot) {
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

function maze_trialsLoopBegin(maze_trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    maze_trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 10, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'maze_trials'
    });
    psychoJS.experiment.addLoop(maze_trials); // add the loop to the experiment
    currentLoop = maze_trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisMaze_trial of maze_trials) {
      snapshot = maze_trials.getSnapshot();
      maze_trialsLoopScheduler.add(importConditions(snapshot));
      maze_trialsLoopScheduler.add(MazeRoutineRoutineBegin(snapshot));
      maze_trialsLoopScheduler.add(MazeRoutineRoutineEachFrame());
      maze_trialsLoopScheduler.add(MazeRoutineRoutineEnd(snapshot));
      maze_trialsLoopScheduler.add(maze_trialsLoopEndIteration(maze_trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function maze_trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(maze_trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function maze_trialsLoopEndIteration(scheduler, snapshot) {
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

function questionnaireRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'questionnaire' ---
    t = 0;
    questionnaireClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    questionnaireMaxDurationReached = false;
    // update component parameters for each repeat
    questionsimages.setImage(questions_images);
    slider.reset()
    next.keys = undefined;
    next.rt = undefined;
    _next_allKeys = [];
    // setup some python lists for storing info about the mouse
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('questionnaire.started', globalClock.getTime());
    questionnaireMaxDuration = null
    // keep track of which components have finished
    questionnaireComponents = [];
    questionnaireComponents.push(questionsimages);
    questionnaireComponents.push(slider);
    questionnaireComponents.push(next);
    questionnaireComponents.push(mouse);
    
    for (const thisComponent of questionnaireComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function questionnaireRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'questionnaire' ---
    // get current time
    t = questionnaireClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *questionsimages* updates
    if (t >= 0.0 && questionsimages.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      questionsimages.tStart = t;  // (not accounting for frame time here)
      questionsimages.frameNStart = frameN;  // exact frame index
      
      questionsimages.setAutoDraw(true);
    }
    
    
    // *slider* updates
    if (t >= 0.0 && slider.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      slider.tStart = t;  // (not accounting for frame time here)
      slider.frameNStart = frameN;  // exact frame index
      
      slider.setAutoDraw(true);
    }
    
    
    // *next* updates
    if (t >= 0.0 && next.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      next.tStart = t;  // (not accounting for frame time here)
      next.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { next.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { next.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { next.clearEvents(); });
    }
    
    if (next.status === PsychoJS.Status.STARTED) {
      let theseKeys = next.getKeys({keyList: ['right'], waitRelease: false});
      _next_allKeys = _next_allKeys.concat(theseKeys);
      if (_next_allKeys.length > 0) {
        next.keys = _next_allKeys[_next_allKeys.length - 1].name;  // just the last key pressed
        next.rt = _next_allKeys[_next_allKeys.length - 1].rt;
        next.duration = _next_allKeys[_next_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
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
    for (const thisComponent of questionnaireComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function questionnaireRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'questionnaire' ---
    for (const thisComponent of questionnaireComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('questionnaire.stopped', globalClock.getTime());
    psychoJS.experiment.addData('slider.response', slider.getRating());
    psychoJS.experiment.addData('slider.rt', slider.getRT());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(next.corr, level);
    }
    psychoJS.experiment.addData('next.keys', next.keys);
    if (typeof next.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('next.rt', next.rt);
        psychoJS.experiment.addData('next.duration', next.duration);
        routineTimer.reset();
        }
    
    next.stop();
    // store data for psychoJS.experiment (ExperimentHandler)
    // the Routine "questionnaire" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function wait_for_mazeRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'wait_for_maze' ---
    t = 0;
    wait_for_mazeClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    wait_for_mazeMaxDurationReached = false;
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    psychoJS.experiment.addData('wait_for_maze.started', globalClock.getTime());
    wait_for_mazeMaxDuration = null
    // keep track of which components have finished
    wait_for_mazeComponents = [];
    wait_for_mazeComponents.push(maze_explanation);
    wait_for_mazeComponents.push(key_resp);
    
    for (const thisComponent of wait_for_mazeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function wait_for_mazeRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'wait_for_maze' ---
    // get current time
    t = wait_for_mazeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *maze_explanation* updates
    if (t >= 0.0 && maze_explanation.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      maze_explanation.tStart = t;  // (not accounting for frame time here)
      maze_explanation.frameNStart = frameN;  // exact frame index
      
      maze_explanation.setAutoDraw(true);
    }
    
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }
    
    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        key_resp.duration = _key_resp_allKeys[_key_resp_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
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
    for (const thisComponent of wait_for_mazeComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function wait_for_mazeRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'wait_for_maze' ---
    for (const thisComponent of wait_for_mazeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('wait_for_maze.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        psychoJS.experiment.addData('key_resp.duration', key_resp.duration);
        routineTimer.reset();
        }
    
    key_resp.stop();
    // the Routine "wait_for_maze" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function wait_a_secRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'wait_a_sec' ---
    t = 0;
    wait_a_secClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(0.500000);
    wait_a_secMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('wait_a_sec.started', globalClock.getTime());
    wait_a_secMaxDuration = null
    // keep track of which components have finished
    wait_a_secComponents = [];
    wait_a_secComponents.push(loading);
    
    for (const thisComponent of wait_a_secComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function wait_a_secRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'wait_a_sec' ---
    // get current time
    t = wait_a_secClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *loading* updates
    if (t >= 0.0 && loading.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      loading.tStart = t;  // (not accounting for frame time here)
      loading.frameNStart = frameN;  // exact frame index
      
      loading.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (loading.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      loading.setAutoDraw(false);
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
    for (const thisComponent of wait_a_secComponents)
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

function wait_a_secRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'wait_a_sec' ---
    for (const thisComponent of wait_a_secComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('wait_a_sec.stopped', globalClock.getTime());
    if (wait_a_secMaxDurationReached) {
        routineTimer.add(wait_a_secMaxDuration);
    } else {
        routineTimer.add(-0.500000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function MazeRoutineRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'MazeRoutine' ---
    t = 0;
    MazeRoutineClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    MazeRoutineMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from Maze_Code
    console.log(continueRoutine);
    WALL_COLOR = [(- 0.4), (- 0.4), (- 0.4)];
    PATH_COLOR = [0.7, 0.7, 0.7];
    AGENT_COLOR = [(- 1), (- 1), 1];
    GOAL_COLOR = [1, (- 1), (- 1)];
    maz = new Maze({"size": 10});
    maz.generate_maze();
    cell_size = (0.4 / maz.size);
    player_size = 0.9;
    pressed_keys = [];
    maze_cells = maz.draw_maze();
    for (var cell, _pj_c = 0, _pj_a = maze_cells, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
        cell = _pj_a[_pj_c];
        cell.draw();
    }
    Timer = new visual.TextStim({"win": psychoJS.window, "name": "Timer", "text": "0", "font": "Arial", "pos": [0, 0.45], "draggable": false, "height": 0.05, "wrapWidth": null, "ori": 0.0, "color": "white", "colorSpace": "rgb", "opacity": null, "languageStyle": "LTR", "depth": (- 1.0)});
    
    psychoJS.experiment.addData('MazeRoutine.started', globalClock.getTime());
    // keep track of which components have finished
    MazeRoutineComponents = [];
    MazeRoutineComponents.push(Maze_timer);
    
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
    // is it time to end the Routine? (based on condition)
    if (Boolean(skip_routine)) {
        continueRoutine = false
    }
    // Run 'Each Frame' code from Maze_Code
    t = routineTimer.getTime();
    if ((! maz.goal_reached)) {
        keys = psychoJS.eventManager.getKeys();
        for (var key, _pj_c = 0, _pj_a = keys, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
            key = _pj_a[_pj_c];
            maz.move_agent(key, maz.agent_location);
        }
        for (var cell, _pj_c = 0, _pj_a = maze_cells, _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
            cell = _pj_a[_pj_c];
            cell.draw();
        }
        maz.draw_agent();
        Timer.setText(util.round(t).toString(), {"log": false});
        Timer.draw();
        continueRoutine = true;
    } else {
    }
    
    
    // *Maze_timer* updates
    if (t >= 0.0 && Maze_timer.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Maze_timer.tStart = t;  // (not accounting for frame time here)
      Maze_timer.frameNStart = frameN;  // exact frame index
      
      Maze_timer.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 25 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (Maze_timer.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      Maze_timer.setAutoDraw(false);
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
    if (continueRoutine) {
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
    console.log("routine over", continueRoutine);
    
    // the Routine "MazeRoutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
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
