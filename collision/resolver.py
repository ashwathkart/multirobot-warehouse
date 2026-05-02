def exchange2x2(a0, b0, adest, bdest):
    """Given a and b within a 2x2 cell with no other obstacles, returns
    a sequence of moves that will get a from position a0 to adest and b
    from position b0 to bdest.

    The sequence is a list in the form ['au','bl*','ar'] where the first
    character is the object to move, and the second character is the direction
    (u: up, d: down, r: right, l: left)
    If the third character * is given, then this means it can be done at the same time as the
    previous action
    """
    assert a0 != b0, "Objects have the same start?"
    assert adest != bdest, "Objects have the same destination?"
    xlow = min(a0[0], b0[0], adest[0], bdest[0])
    assert max(a0[0], b0[0], adest[0], bdest[0]) <= xlow + 1, "Objects aren't in a 2x2 box?"
    ylow = min(a0[1], b0[1], adest[1], bdest[1])
    assert max(a0[1], b0[1], adest[1], bdest[1]) <= ylow + 1, "Objects aren't in a 2x2 box?"
    a0 = (a0[0] - xlow, a0[1] - ylow)
    b0 = (b0[0] - xlow, b0[1] - ylow)
    adest = (adest[0] - xlow, adest[1] - ylow)
    bdest = (bdest[0] - xlow, bdest[1] - ylow)
    path = []
    while a0 != adest or b0 != bdest:
        aopts = []
        aacts = []
        bopts = []
        bacts = []
        if adest[0] > a0[0]:
            aopts.append((1, a0[1]))
            aacts.append('r')
        elif adest[0] < a0[0]:
            aopts.append((0, a0[1]))
            aacts.append('l')
        if adest[1] > a0[1]:
            aopts.append((a0[0], 1))
            aacts.append('u')
        elif adest[1] < a0[1]:
            aopts.append((a0[0], 0))
            aacts.append('d')
        if bdest[0] > b0[0]:
            bopts.append((1, b0[1]))
            bacts.append('r')
        elif bdest[0] < b0[0]:
            bopts.append((0, b0[1]))
            bacts.append('l')
        if bdest[1] > b0[1]:
            bopts.append((b0[0], 1))
            bacts.append('u')
        elif bdest[1] < b0[1]:
            bopts.append((b0[0], 0))
            bacts.append('d')
        moved = False
        for i in range(len(aopts)):
            if aopts[i] != b0:
                path.append('a' + aacts[i])
                a0 = aopts[i]
                moved = True
                break
        for i in range(len(bopts)):
            if bopts[i] != a0:
                path.append('b' + bacts[i])
                if moved:
                    path[-1] = path[-1] + '*'
                b0 = bopts[i]
                moved = True
                break
        if not moved:
            if a0 == adest and b0 == bdest:
                break
            else:
                if a0[0] == b0[0]:
                    if b0[0] == 0:
                        b0 = (1, b0[1])
                        path.append('br')
                    else:
                        b0 = (0, b0[1])
                        path.append('bl')
                else:
                    assert a0[1] == b0[1]
                    if b0[1] == 0:
                        b0 = (b0[0], 1)
                        path.append('bu')
                    else:
                        b0 = (b0[0], 0)
                        path.append('bd')
    return path


def exchangeToMoves(soln: list, robots: list, simultaneous=True) -> list:
    """From an exchange2x2 solution and a list of robot integers, returns
    a list of moves.  If simultaneous=True, these are simultaneous moves
    in the form [moves1,moves2,...] with each element
    movesN = [(robot1,move1),...,(robotk,movek)] giving the moves performed
    simultaneously."""
    if simultaneous:
        moves = []
        lastmove = []
        for mv in soln:
            if mv[-1] != '*' and len(lastmove) > 0:
                moves.append(lastmove)
                lastmove = []
            lastmove.append([robots[0] if mv[0] == 'a' else robots[1], mv[1]])
        moves.append(lastmove)
    else:
        moves = []
        for mv in soln:
            moves.append([robots[0] if mv[0] == 'a' else robots[1], mv[1]])
    return moves
