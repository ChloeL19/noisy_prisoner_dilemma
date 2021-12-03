#!/usr/bin/python

# TODO:
# - instructions to compete.php
# - prettify a bit -- center columns in table, maybe add some background colors.  Put the score up front.  
# - write script to run tournament with lots of agents....


#print "Running!"

import sys
import random
import re

# Constants:
MAX_STATES = 5
CC = 0
CD = 1
DC = 2
DD = 3
MAX_ROUNDS = 1000
ERROR_PROB = 0.05
USAGE_MSG = """USAGE: pd.py numrounds [html] [debug]

numrounds must be a positive int

if html is specified, outputs html (for use from php script)
if debug is specified, outputs more debugging info

The program then expects up to 5 lines of the form:
0: 0.1 1 0 0 1
1: 0.7 1 1 0 2

a blank line, and up to 5 more lines of the same form.

See the documentation for details of what the agent format must be.
"""

def usage():
    #    sys.stderr.write(USAGE_MSG)
    print(USAGE_MSG)
    sys.exit()

def stringify(lst):
    return map(str, lst)

class State:
    def __init__(self, num, prob_coop, on_cc, on_cd, on_dc, on_dd):
        self.num = num
        if prob_coop < 0.0 or prob_coop > 1.000001:
            raise Exception("prob_coop %f for state %d must be in [0,1]" % (
                prob_coop, num))
        self.prob_coop = prob_coop
        self.on_cc = on_cc
        self.on_cd = on_cd
        self.on_dc = on_dc
        self.on_dd = on_dd
        
    def check(self, n):
        """
        Given that states 0, 1, ..., n-1 exist, check that all params of this 
        state are valid
        """
        if self.num >= n:
            raise Exception("invalid state number %d" % self.num)
        if self.on_cc >= n:
            raise Exception("invalid jump from state %d on on_cc: %d" % (
                self.num, self.on_cc))
        if self.on_cd >= n:
            raise Exception("invalid jump from state %d on on_cc: %d" % (
                self.num, self.on_cd))
        if self.on_dc >= n:
            raise Exception("invalid jump from state %d on on_cc: %d" % (
                self.num, self.on_dc))
        if self.on_dd >= n:
            raise Exception("invalid jump from state %d on on_cc: %d" % (
                self.num, self.on_dd))

    def __repr__(self):
        return "State(id=%d, prob_c=%0.2f, on_cc=%d, on_cd=%d, on_dc=%d, on_dd=%d)" % (
            self.num, self.prob_coop, self.on_cc, self.on_cd, self.on_dc, self.on_dd)
    


class Player:
    def __init__(self, id, lines):
        self.id = id
        self.current_state = 0
        self.states = []
        pattern = r"^\d:\s+[01]\.\d+(\s+\d){4}"
        self.strategyHtml = "[<br>"
        k = 0  # Which state we're expecting to see next
        for line in lines:
            if len(line) > 0 and line[0] == '#':
                continue  # skip comments
            if not re.match(pattern, line):
                raise Exception("Line does not have the right pattern: '%s'" % line)
            (num, prob, on1, on2, on3, on4) = line.split()
            num = int(num[:-1])  # pull off the ':'
            if k != num:
                raise Exception("Was expecting state %d. Saw %d" % (k,num))
            k += 1
            prob = float(prob)
            on_cc = int(on1)
            on_cd = int(on2)
            on_dc = int(on3)
            on_dd = int(on4)
            state = State(num, prob, on_cc, on_cd, on_dc, on_dd)
            self.strategyHtml+= "%s%d, %.1f, %d, %d, %d, %d<br>" %("&nbsp;" * 10,num,prob, on_cc, on_cd, on_dc, on_dd)
            self.states.append(state)
            self.name = "automaton"

        
        n = len(self.states)
        self.strategyHtml += "]"
        if n < 1 or n > MAX_STATES:
            raise Exception("Must have between 1 and %d states. Counted %d instead." %
                            (MAX_STATES, n))
        for state in self.states:
            state.check(n)


    def __repr__(self):
        return "Player(id=%d, states=%s)" % (self.id, "\n".join(stringify(self.states)))
    def getStrategy(self):
        return self.strategyHtml
        
    def act(self):
        """ Cooperate or defect """
        if random.random() < self.states[self.current_state].prob_coop:
            return 'C'
        else:
            return 'D'
        # takes the output of neural network, samples from the prob to give action
        # the get_action function

    def react(self, result):
        """ Transition based on the joint action (possibly noisy) """
        cur = self.states[self.current_state]
        if result == CC:
            self.current_state = cur.on_cc
        elif result == CD:
            self.current_state = cur.on_cd
        elif result == DC:
            self.current_state = cur.on_dc
        elif result == DD:
            self.current_state = cur.on_dd
        # update the state of RL agent
    
    def update(self, train=None, timestep_reward=None):
        # for the automaton, updating does nothing
        pass
    
def readplayer(id, f=sys.stdin):
    lines = []
    # Read until we get EOF or blank line
    while True:
        line = f.readline()
        if not line: break
        if re.match(r"^\s*$", line):
            break
        lines.append(line)
    # now actually create the player
    return Player(id, lines)

def play(p1, p2, numrounds, debug_flag, html, print_stuff=True, trainbool=False):
    """
    print_stuff added so that tournament can run without printing every single game...
    """
    def noisify(a):
        if random.random() < ERROR_PROB:
            if a == "C":
                return "D"
            else:
                return "C"
        else:
            return a
                
    def debug(s):
        if debug_flag:
            print(s)

    def myprint(s):
        if print_stuff:
            print(s)

    def result(a1, a2):
        if a1 == 'C' and a2 == 'C':
            return CC
        if a1 == 'C' and a2 == 'D':
            return CD
        if a1 == 'D' and a2 == 'C':
            return DC
        if a1 == 'D' and a2 == 'D':
            return DD
        raise Exception("Unknown results: %s, %s" % (a1, a2))

    def header():
        if html:
            myprint("""<table>
<tr>
   <th>Round</th>
   <th>Actions</th>
   <th>Observed as</th>
   <th>Scores</th>
   <th>New states</th>
</tr>
""")
        else:
            debug("Starting with players:\n%s\n%s" % (p1, p2))

    def footer(score1, score2):
        if html:
            myprint("</table>")
            myprint("<h3>Final scores: (%d, %d)</h3>" % (score1, score2))
        else:
            myprint("Final scores (%d, %d)" % (score1, score2))


    def log_round(r, a1, a2, o_a1, o_a2, s1, s2, ns1, ns2):
        if html:
            myprint("""<tr>
   <td>%d</td>
   <td>%s, %s</td>
   <td>%s, %s</td>
   <td>%d, %d</td>
   <td>%d, %d</td>
</tr>""" % (r, a1, a2, o_a1, o_a2, s1, s2, ns1, ns2))
        else:
            myprint("Round %d: actions (%s, %s). Observed as (%s, %s). Scores (%d, %d), " % (
            r, a1, a2, observed_a1, observed_a2, s1, s2))
            
    
    header()

    score1 = 0
    score2 = 0
    for r in range(numrounds):
        a1 = p1.act()
        a2 = p2.act()

        # if (p1.name == "RL" or p2.name == "RL"):
        #     import pdb; pdb.set_trace();

        observed_a1 = noisify(a1)
        observed_a2 = noisify(a2)

        # if (p1.name == "RL" or p2.name == "RL"):
        #     import pdb; pdb.set_trace();

        s1 = s2 = 0
        if a1 == 'C' and a2 == 'C':
            s1 = s2 = 3
        if a1 == 'C' and a2 == 'D':
            s2 = 5
        if a1 == 'D' and a2 == 'C':
            s1 = 5
        if a1 == 'D' and a2 == 'D':
            s1 = s2 = 1


        score1 += s1
        score2 += s2
        #print("About to update agent1.")
        loss1 = p1.update(train=trainbool, timestep_reward=s1)
        #print("About to update agent2.")
        loss2 = p2.update(train=trainbool, timestep_reward=s2)
        if (p1.name == "RL" or p2.name=="RL"):
            print("Round: {}, Loss 1: {}, loss 2: {}\n".format(r, loss1, loss2))
        #print("Updated agent 2.")
        #print("Updating state of p1")
        p1.react(result(a1, observed_a2))
        #print("Updating state of p2")
        p2.react(result(a2, observed_a1))
        #print("Updated state of p2")
        log_round(r, a1, a2, observed_a1, observed_a2, s1, s2, p1.current_state, p2.current_state)
        #print("Logged the round.")
    footer(score1, score2)
    return (score1, score2)
        

def main(args):
    import pdb; pdb.set_trace();
    if len(args) < 2 or len(args) > 4:
        usage()

    try:
        numrounds = int(args[1])
        if numrounds < 0:
            raise Exception("numrounds must be positive")
        if numrounds > MAX_ROUNDS:
            raise Exception("numrounds can't be greater than %d" % MAX_ROUNDS)
        debug = False
        html = False
        if "debug" in args:
            debug = True
        if "html" in args:
            html = True
    except Exception as e:
        print("Bad argument:{}".format(e))
        usage()
            
    p1 = readplayer(1)
    p2 = readplayer(2)

    play(p1, p2, numrounds, debug, html)

if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception as e:
        print("ERROR: {}".format(e))
        print("")
        print(sys.exc_info())
