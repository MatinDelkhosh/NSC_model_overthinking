﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1post4),
    on November 01, 2024, at 19:46
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from Maze_Code


# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1post4'
expName = 'OverthinkingModel'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 720]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\Matin\\stuff\\NSC\\code\\NSC_model_overthinking\\Online Task\\PsychoPy project\\OverthinkingModel_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0.5000, 0.5000, 0.5000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0.5000, 0.5000, 0.5000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Questionnaire" ---
    win.allowStencil = True
    form = visual.Form(win=win, name='form',
        items='D:/Matin/stuff/NSC/code/NSC_model_overthinking/Online Task/questionnaire/questionnaire.xlsx',
        textHeight=0.03,
        font='Open Sans',
        randomize=False,
        style='dark',
        fillColor=None, borderColor=None, itemColor='white', 
        responseColor='white', markerColor='red', colorSpace='rgb', 
        size=(1.2, .8),
        pos=(0, .05),
        itemPadding=0.05,
        depth=0
    )
    
    # --- Initialize components for Routine "MazeRoutine" ---
    # Run 'Begin Experiment' code from Maze_Code
    import random
    
    # Colors
    WALL_COLOR = (-.4, -.4, -.4)  # Dark Gray
    PATH_COLOR = (.7, .7, .7)     # white
    AGENT_COLOR = (-1, -1, 1)  # blue
    GOAL_COLOR = (1, -1, -1)   # red
    
    class Maze:
        def __init__(self, size=8):
            self.size = size
            self.maze = [[1] * (2 * size + 1) for _ in range(2 * size + 1)]
            self.visited = set([])
            self.goal_location = None
            self.agent_location = None 
            self.walls = set([])
            self.goal_reached = False
    
        def generate_maze(self):
            start = (random.randint(0, self.size - 1) * 2 + 1, random.randint(0, self.size - 1) * 2 + 1)
            self.maze[start[1]][start[0]] = 0
            self.walk_maze(start)
            
            for x in range(1, self.size * 2, 2):
                for y in range(1, self.size * 2, 2):
                    self.walls.add((x, y))
    
            for _ in range(4):
                if not self.walls:
                    break
                wall = random.choice(list(self.walls))
                self.maze[wall[1]][wall[0]] = 0
                self.walls.remove(wall)
                
            self.goal_location = [(self.size - 1) * 2 + 1, (self.size - 1) * 2 + 1]
    
            while True:
                agent_start = [1, 1]
                if agent_start != self.goal_location:
                    self.agent_location = agent_start
                    break
    
            return self.maze
            
        def walk_maze(self, s):
            self.visited.add(s)
            neighbors = self.neighbors(s)
            random.shuffle(neighbors)
    
            for n in neighbors:
                if n not in self.visited:
                    self.remove_wall(s, n)
                    self.walk_maze(n)
                    
        def neighbors(self, s):
            x, y = s
            potential_neighbors = [(x-2, y), (x+2, y), (x, y-2), (x, y+2)]
    
            neighbors = []
            for n in potential_neighbors:
                nx, ny = n
                if 0 <= nx < self.size * 2 + 1 and 0 <= ny < self.size * 2 + 1:
                        neighbors.append((nx, ny))
                        
            return neighbors
            
        def remove_wall(self, s, n):
            sx, sy = s
            nx, ny = n
            
            if sx == nx:  # Vertical neighbors
                wall_pos = (sx, min(sy, ny) + 1)
                self.maze[wall_pos[1]][wall_pos[0]] = 0  # Remove horizontal wall
            elif sy == ny:  # Horizontal neighbors
                wall_pos = (min(sx, nx) + 1, sy)
                self.maze[wall_pos[1]][wall_pos[0]] = 0  # Remove vertical wall
    
            self.maze[ny][nx] = 0  # Mark the new cell as a path
            self.walls.discard(wall_pos)
    
        def teleport_agent(self):
            # Teleport agent to a random location, excluding the goal location
            while True:
                random_location = (random.randint(0, self.size - 1) * 2 + 1, random.randint(0, self.size - 1) * 2 + 1)
                if random_location != self.goal_location:
                    return random_location
                
        def move_agent(self, key, pos):
            new_pos = pos[:]
            if key == 'up' or key == 'w':
                new_pos[1] += 1
            elif key == 'down' or key == 's':
                new_pos[1] -= 1
            elif key == 'left' or key == 'a':
                new_pos[0] -= 1
            elif key == 'right' or key == 'd':
                new_pos[0] += 1
    
            c1 = 0 <= new_pos[1] < self.size*2
            c2 = 0 <= new_pos[0] < self.size*2
            
            if c1 and c2:
                if self.maze[new_pos[1]][new_pos[0]] == 0:
                    self.agent_location = new_pos
                    if self.agent_location == self.goal_location:
                        self.goal_reached = True
            else:
                self.agent_location = pos
                
    
        def draw_maze(self):
            rects = []
            for y in range(len(self.maze)):
                for x in range(len(self.maze[0])):
                    color = WALL_COLOR if self.maze[y][x] == 1 else PATH_COLOR
                    rect = visual.Rect(win, width=cell_size, height=cell_size, fillColor=color, interpolate=True)
                    rect.pos = [x * cell_size - (self.size * cell_size), y * cell_size - (self.size * cell_size)]
                    rects.append(rect)
            return rects
    
        def draw_agent(self):
            # Draw the agent
            ax, ay = self.agent_location
            agent = visual.Circle(win, radius=cell_size * player_size / 2, fillColor=AGENT_COLOR)
            if self.goal_reached:
                agent.fillColor = 'yellow'
            agent.pos = [ax * cell_size - (self.size * cell_size), ay * cell_size - (self.size * cell_size)]
            agent.draw()
    
            # Draw the goal
            if not self.goal_reached:
                gx, gy = self.goal_location
                goal = visual.Rect(win, width=cell_size, height=cell_size, fillColor=GOAL_COLOR)
                goal.pos = [gx * cell_size - (self.size * cell_size), gy * cell_size - (self.size * cell_size)]
                goal.draw()
    Maze_timer = visual.ShapeStim(
        win=win, name='Maze_timer',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=0.0, depth=-1.0, interpolate=True)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Questionnaire" ---
    # create an object to store info about Routine Questionnaire
    Questionnaire = data.Routine(
        name='Questionnaire',
        components=[form],
    )
    Questionnaire.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Questionnaire
    Questionnaire.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Questionnaire.tStart = globalClock.getTime(format='float')
    Questionnaire.status = STARTED
    thisExp.addData('Questionnaire.started', Questionnaire.tStart)
    Questionnaire.maxDuration = None
    # keep track of which components have finished
    QuestionnaireComponents = Questionnaire.components
    for thisComponent in Questionnaire.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Questionnaire" ---
    Questionnaire.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *form* updates
        
        # if form is starting this frame...
        if form.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            form.frameNStart = frameN  # exact frame index
            form.tStart = t  # local t and not account for scr refresh
            form.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(form, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'form.started')
            # update status
            form.status = STARTED
            form.setAutoDraw(True)
        
        # if form is active this frame...
        if form.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Questionnaire.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Questionnaire.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Questionnaire" ---
    for thisComponent in Questionnaire.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Questionnaire
    Questionnaire.tStop = globalClock.getTime(format='float')
    Questionnaire.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Questionnaire.stopped', Questionnaire.tStop)
    form.addDataToExp(thisExp, 'rows')
    form.autodraw = False
    thisExp.nextEntry()
    # the Routine "Questionnaire" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "MazeRoutine" ---
        # create an object to store info about Routine MazeRoutine
        MazeRoutine = data.Routine(
            name='MazeRoutine',
            components=[Maze_timer],
        )
        MazeRoutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from Maze_Code
        # Initialize the maze
        maz = Maze(size=12)
        maz.generate_maze()
        
        # Set the size of the cells
        cell_size = .4/maz.size
        player_size = .9
        
        pressed_keys = []
        
        maze_cells = maz.draw_maze()
        for cell in maze_cells:
            cell.draw()
            
        Timer = visual.TextStim(win=win, name='Timer',
                text='0',
                font='Arial',
                pos=(0, .45), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
                color='white', colorSpace='rgb', opacity=None, 
                languageStyle='LTR',
                depth=-1.0);
                
        Overlay = visual.Rect(
                win=win, name='Overlay',
                width=(.9, .9)[0], height=(.5, .5)[1],
                ori=0.0, pos=(0, 0), draggable=False, anchor='center',
                lineWidth=1.0,
                colorSpace='rgb', lineColor='white', fillColor='green',
                opacity=0.6, depth=-2.0, interpolate=True)
                
        Countdown = False
        skip_routine = False
        # store start times for MazeRoutine
        MazeRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        MazeRoutine.tStart = globalClock.getTime(format='float')
        MazeRoutine.status = STARTED
        thisExp.addData('MazeRoutine.started', MazeRoutine.tStart)
        # keep track of which components have finished
        MazeRoutineComponents = MazeRoutine.components
        for thisComponent in MazeRoutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "MazeRoutine" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        MazeRoutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on condition)
            if bool(skip_routine):
                continueRoutine = False
            # Run 'Each Frame' code from Maze_Code
            t = routineTimer.getTime()
            if not maz.goal_reached and t < 20:
                keys = event.getKeys()
                for key in keys:
                    maz.move_agent(key, maz.agent_location)
                    break
                
                for cell in maze_cells:
                    cell.draw()
                maz.draw_agent()
                Timer.setText(str(round(t)), log=False)
                Timer.draw()
                
            else:
                if not Countdown:
                    ts = t
                    Countdown = True
                    
                for cell in maze_cells:
                    cell.draw()
                maz.draw_agent()
                
                if not maz.goal_reached:
                    Overlay.fillColor = 'red'
                    Overlay.opacity = 0.6
                Overlay.draw()
                
                text = 'Success!' if maz.goal_reached else 'Time Over!'
                Timer.setText(text)
                Timer.pos = (0.0)
                Timer.draw()
                
                if t - ts >= 2:
                    skip_routine = True
            
            # *Maze_timer* updates
            
            # if Maze_timer is starting this frame...
            if Maze_timer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Maze_timer.frameNStart = frameN  # exact frame index
                Maze_timer.tStart = t  # local t and not account for scr refresh
                Maze_timer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Maze_timer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Maze_timer.started')
                # update status
                Maze_timer.status = STARTED
                Maze_timer.setAutoDraw(True)
            
            # if Maze_timer is active this frame...
            if Maze_timer.status == STARTED:
                # update params
                pass
            
            # if Maze_timer is stopping this frame...
            if Maze_timer.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Maze_timer.tStartRefresh + 25-frameTolerance:
                    # keep track of stop time/frame for later
                    Maze_timer.tStop = t  # not accounting for scr refresh
                    Maze_timer.tStopRefresh = tThisFlipGlobal  # on global time
                    Maze_timer.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Maze_timer.stopped')
                    # update status
                    Maze_timer.status = FINISHED
                    Maze_timer.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                MazeRoutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in MazeRoutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "MazeRoutine" ---
        for thisComponent in MazeRoutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for MazeRoutine
        MazeRoutine.tStop = globalClock.getTime(format='float')
        MazeRoutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('MazeRoutine.stopped', MazeRoutine.tStop)
        # the Routine "MazeRoutine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
