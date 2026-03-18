"""
Flappy Bird — BCI-controlled edition
=====================================

Usage
-----
Live mode (Cyton board on COM3):
    python flappy.py

Live mode (different port):
    python flappy.py --port /dev/ttyUSB0

Offline replay mode (from a recorded .npy file):
    python flappy.py --offline path/to/eeg.npy

Offline replay with matching timestamps:
    python flappy.py --offline path/to/eeg.npy --timestamps path/to/timestamps.npy

Offline replay as fast as possible (no real-time pacing):
    python flappy.py --offline path/to/eeg.npy --no-realtime

BrainFlow synthetic board (no hardware, requires brainflow installed):
    python flappy.py --synthetic

Keyboard only (BCI disabled):
    python flappy.py --no-bci

Adjust blink detection sensitivity:
    python flappy.py --threshold 0.6

The EEG .npy file must have shape (n_channels, n_samples) with at least 2 channels.
"""

from itertools import cycle
import argparse
import random
import sys

import pygame
from pygame.locals import *

from bci_controller import make_bci_controller
from bci_controller import BCIController


FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79

IMAGES, SOUNDS, HITMASKS = {}, {}, {}

PLAYERS_LIST = (
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

try:
    xrange
except NameError:
    xrange = range


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BCI-controlled Flappy Bird")
    parser.add_argument(
        "--offline",
        metavar="EEG_NPY",
        default=None,
        help="Path to a .npy EEG file (shape: n_channels x n_samples) for offline replay.",
    )
    parser.add_argument(
        "--timestamps",
        metavar="TS_NPY",
        default=None,
        help="Path to a matching timestamps .npy file (1-D, seconds). Optional.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Process offline file as fast as possible instead of real-time replay.",
    )
    parser.add_argument(
        "--port",
        default="COM3",
        help="Serial port for live Cyton board (default: COM3).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use BrainFlow synthetic board (no hardware needed).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Blink detection probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--model",
        default="model.joblib",
        help="Path to trained model file (default: blink_model.joblib).",
    )
    parser.add_argument(
        "--no-bci",
        action="store_true",
        help="Disable BCI entirely (keyboard-only mode).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird — BCI')

    bci_controller = make_bci_controller(
        offline_eeg_path=args.offline,
        offline_timestamps_path=args.timestamps,
        serial_port=args.port,
        use_synthetic_board=args.synthetic,
        realtime_replay=not args.no_realtime,
        blink_threshold=args.threshold,
        model_path=args.model,
        enabled=not args.no_bci,
    )
    bci_controller.start()

    # number sprites
    IMAGES['numbers'] = tuple(
        pygame.image.load(f'assets/sprites/{i}.png').convert_alpha()
        for i in range(10)
    )

    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    IMAGES['message']  = pygame.image.load('assets/sprites/message.png').convert_alpha()
    IMAGES['base']     = pygame.image.load('assets/sprites/base.png').convert_alpha()

    soundExt = '.wav' if 'win' in sys.platform else '.ogg'
    for name in ('die', 'hit', 'point', 'swoosh', 'wing'):
        SOUNDS[name] = pygame.mixer.Sound(f'assets/audio/{name}{soundExt}')

    try:
        while True:
            IMAGES['background'] = pygame.image.load(
                BACKGROUNDS_LIST[random.randint(0, len(BACKGROUNDS_LIST) - 1)]
            ).convert()

            randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
            IMAGES['player'] = tuple(
                pygame.image.load(p).convert_alpha() for p in PLAYERS_LIST[randPlayer]
            )

            pipeindex = random.randint(0, len(PIPES_LIST) - 1)
            IMAGES['pipe'] = (
                pygame.transform.flip(
                    pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True
                ),
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
            )

            HITMASKS['pipe'] = (getHitmask(IMAGES['pipe'][0]), getHitmask(IMAGES['pipe'][1]))
            HITMASKS['player'] = tuple(getHitmask(IMAGES['player'][i]) for i in range(3))

            movementInfo = showWelcomeAnimation(bci_controller)
            crashInfo    = mainGame(movementInfo, bci_controller)
            showGameOverScreen(crashInfo, bci_controller)
    finally:
        bci_controller.stop()


# ---------------------------------------------------------------------------
# Game screens
# ---------------------------------------------------------------------------

def trigger_flap(playery, playerFlapAcc):
    if playery > -2 * IMAGES['player'][0].get_height():
        SOUNDS['wing'].play()
        return playerFlapAcc, True
    return None, False


def showWelcomeAnimation(bci_controller):
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)
    basex = 0
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        start_requested = False

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                start_requested = True

        if bci_controller.should_jump():
            start_requested = True

        if start_requested:
            SOUNDS['wing'].play()
            return {
                'playery': playery + playerShmVals['val'],
                'basex': basex,
                'playerIndexGen': playerIndexGen,
            }

        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(movementInfo, bci_controller):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    upperPipes = [
        {'x': SCREENWIDTH + 200,                  'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + SCREENWIDTH / 2, 'y': newPipe2[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200,                  'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + SCREENWIDTH / 2, 'y': newPipe2[1]['y']},
    ]

    dt = FPSCLOCK.tick(FPS) / 1000
    pipeVelX = -128 * dt

    playerVelY    = -9
    playerMaxVelY = 10
    playerAccY    = 1
    playerRot     = 45
    playerVelRot  = 3
    playerRotThr  = 20
    playerFlapAcc = -9
    playerFlapped = False

    while True:
        flap_requested = False

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                flap_requested = True

        if bci_controller.should_jump():
            flap_requested = True

        if flap_requested:
            newVelY, did_flap = trigger_flap(playery, playerFlapAcc)
            if did_flap:
                playerVelY = newVelY
                playerFlapped = True

        crashTest = checkCrash(
            {'x': playerx, 'y': playery, 'index': playerIndex},
            upperPipes, lowerPipes,
        )
        if crashTest[0]:
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot,
            }

        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                SOUNDS['point'].play()

        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        if playerRot > -90:
            playerRot -= playerVelRot

        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        if 0 < len(upperPipes) < 3 and 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        if upperPipes and upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        visibleRot = playerRotThr if playerRot > playerRotThr else playerRot
        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(crashInfo, bci_controller):
    score       = crashInfo['score']
    playerx     = SCREENWIDTH * 0.2
    playery     = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY  = crashInfo['playerVelY']
    playerAccY  = 2
    playerRot   = crashInfo['playerRot']
    playerVelRot = 7
    basex       = crashInfo['basex']
    upperPipes  = crashInfo['upperPipes']
    lowerPipes  = crashInfo['lowerPipes']

    SOUNDS['hit'].play()
    if not crashInfo['groundCrash']:
        SOUNDS['die'].play()

    while True:
        restart_requested = False

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                restart_requested = True

        if bci_controller.should_jump():
            restart_requested = True

        if restart_requested and playery + playerHeight >= BASEY - 1:
            return

        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        if playerVelY < 15:
            playerVelY += playerAccY

        if not crashInfo['groundCrash'] and playerRot > -90:
            playerRot -= playerVelRot

        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)
        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx, playery))
        SCREEN.blit(IMAGES['gameover'], (50, 180))
        FPSCLOCK.tick(FPS)
        pygame.display.update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def playerShm(playerShmVals):
    if abs(playerShmVals['val']) == 8:
        playerShmVals['dir'] *= -1
    playerShmVals['val'] += playerShmVals['dir']


def getRandomPipe():
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10
    return [
        {'x': pipeX, 'y': gapY - pipeHeight},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def showScore(score):
    scoreDigits = [int(x) for x in str(score)]
    totalWidth  = sum(IMAGES['numbers'][d].get_width() for d in scoreDigits)
    Xoffset = (SCREENWIDTH - totalWidth) / 2
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]

    playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
    pipeW = IMAGES['pipe'][0].get_width()
    pipeH = IMAGES['pipe'][0].get_height()

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

        pHitMask = HITMASKS['player'][pi]
        uHitmask = HITMASKS['pipe'][0]
        lHitmask = HITMASKS['pipe'][1]

        if pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask):
            return [True, False]
        if pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask):
            return [True, False]

    return [False, False]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def getHitmask(image):
    mask = []
    for x in xrange(image.get_width()):
        mask.append([bool(image.get_at((x, y))[3]) for y in xrange(image.get_height())])
    return mask


if __name__ == '__main__':
    main()
