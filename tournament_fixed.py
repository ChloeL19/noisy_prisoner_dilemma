#!/usr/bin/python

USAGE_MSG = """
USAGE: tournament.py numrounds iters agent_dir [debug] [html] [redact]
Looks for files named *.agent in agent_dir.
Loads them, runs an all pairs tournament, with each pair run iters times for numrounds rounds, prints final average scores.
"""

import sys
import glob
import os
import os.path
import datetime

from pd import play, readplayer
import pd
from RL_tourney import RL_agent

def usage():
    print(USAGE_MSG)
    sys.exit()

do_redact = False

def redact(s):
    if redact:
        try:
            n = len(s)
        except:
            # if this is a number, means this is the RL agent
            return "RL agent"
        keep = 3
        return s[:keep] + '*' * (n-keep)
    else:
        return s


def main(args):
    try:
        if len(args) < 4 or len(args) > 10:
            usage()
    except:
        import pdb; pdb.set_trace();

    try:
        numrounds = int(args[1])
        if numrounds < 0:
            raise Exception("numrounds must be positive")
        if numrounds > pd.MAX_ROUNDS:
            raise Exception("numrounds can't be greater than %d" % pd.MAX_ROUNDS)
        iters = int(args[2])
        if iters < 0:
            raise Exception("iters must be positive")
        agent_dir = args[3]

        debug = False
        html = False
        train = False
        test = False # testing phase of RL agent
        if "debug" in args:
            debug = True
        if "html" in args:
            html = True
        if "redact" in args:
            do_redact = True
        if "train" in args:
            train = True
        if "test" in args:
            test = True
    except Exception as e:
        print("Bad argument: {}".format(e))
        usage()

    def ifhtml(msg):
        if html:
            print(msg)

    ifhtml("<center>")
    ifhtml("<h1>All Pairs Tournament</h1>")

    agent_files = glob.glob(agent_dir + "/*.agent")
    agents = dict()
    n = 0
    for fname in agent_files:
        ifhtml("<p>")
        #if not html:
        #print "Opening %s..." % fname
        with open(fname) as f:
            try:
                name = os.path.basename(fname[:-6])
                p = readplayer(n, f, name)
                n += 1
                # name = os.path.basename(fname[:-6])
                agents[name] = p
                #p.name = name
            except Exception as e:
                ifhtml("<p>")
                print("Couldn't read %s. Skipping. Error: %s" % (fname, e))

    def header(n1, n2):
        n1 = redact(n1)
        n2 = redact(n2)
        #print "Running %s vs %s" % (n1, n2)

    def log_scores(s1, s2):
        if html:
            print("%d, %d<br>" % (s1, s2))
        else:
            print("Scores: %d, %d" % (s1, s2))
    
    def cmp(a, b):
        '''
        Re-defining for Python 3
        '''
        return (a > b) - (a < b)

    # currently hardcoded to include just one RL agent 
    #import pdb; pdb.set_trace();
    rl_agent = RL_agent(initial_coop=0.5, test=test)
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        agents[len(agents)+2] = rl_agent
    except:
        import pdb; pdb.set_trace();
    scores = dict()
    names = [n for n in agents.keys()]
    for k in agents.keys():
        p = agents[k]

    ep_id = 0 # episode ID (for training purposes)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            #import pdb; pdb.set_trace();
            n1 = names[i]
            n2 = names[j]
            header(n1,n2)
            for k in range(iters):
                # RESET AGENTS STATEST TO BE 0 (used to be broken)
                agents[n1].current_state = 0
                agents[n2].current_state = 0
    
                (s1, s2) = play(agents[n1], agents[n2], numrounds, debug, html, curr_time, ep_id, debug, train, test)
                #log_scores(s1, s2)
                scores[n1] = scores.get(n1, 0) + s1
                scores[n2] = scores.get(n2, 0) + s2
            ep_id += 1

    results = scores.items()
    # Sort in descending order by score
    results = sorted([r for r in results], key=lambda res: res[1], reverse=True)
    # n - 1 matches, each with iters * numrounds rounds.
    try:
        norm = float(iters * numrounds * (len(names)-1))
    except:
        import pdb; pdb.set_trace();
    if html:
        print ("<hr><h1>Results</h1>")
        print ("<h3>Average scores per round:</h3>")
        print ("<table>")
        print ("\n".join("<tr><td>%s</td> <td>%.2f</td><td>%s</td></tr>" % (redact(name),score / norm, agents[name].getStrategy()) for (name,score) in results))
        print ("</table>")
    else:
        print("Average scores per round:")
        print ("\n".join("%s: %.2f" % (
            redact(name),score / norm) for (name,score) in results))
    ifhtml("</center>")
    # save the RL model to designated folder
    if not test:
        rl_agent.save_model()

if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception as e:
        print("ERROR: {}".format(e))
        print("")
        print(sys.exc_info())
