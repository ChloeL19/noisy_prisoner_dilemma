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

from pd import play, readplayer
import pd

def usage():
    print USAGE_MSG
    sys.exit()

do_redact = False

def redact(s):
    if redact:
        n = len(s)
        keep = 3
        return s[:keep] + '*' * (n-keep)
    else:
        return s


def main(args):
    if len(args) < 4 or len(args) > 7:
        usage()

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
        if "debug" in args:
            debug = True
        if "html" in args:
            html = True
        if "redact" in args:
            do_redact = True
    except Exception, e:
        print "Bad argument: ", e
        usage()

    def ifhtml(msg):
        if html:
            print msg

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
                p = readplayer(n, f)
                n += 1
                name = os.path.basename(fname[:-6])
                agents[name] = p
            except Exception, e:
                ifhtml("<p>")
                print "Couldn't read %s. Skipping. Error: %s" % (fname, e)

    def header(n1, n2):
        n1 = redact(n1)
        n2 = redact(n2)
        #print "Running %s vs %s" % (n1, n2)

    def log_scores(s1, s2):
        if html:
            print "%d, %d<br>" % (s1, s2)
        else:
            print "Scores: %d, %d" % (s1, s2)

    # CHLOE IDEA: specially load and append the RL agent
    # to the end of the agents dictionary
    # name will be RL1
    # would be cool to investigate adding multiple RLs later

    scores = dict()
    names = agents.keys()
    for k in agents.keys():
        p = agents[k]
        #print "key=", k,"\nvalues=\n", p.getStrategy(),"\n\n"
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            n1 = names[i]
            n2 = names[j]
            header(n1,n2)
            for k in range(iters):
                # RESET AGENTS STATEST TO BE 0 (used to be broken)
                agents[n1].current_state = 0
                agents[n2].current_state = 0
                (s1, s2) = play(agents[n1], agents[n2], numrounds, debug, html, debug)
                #log_scores(s1, s2)
                scores[n1] = scores.get(n1, 0) + s1
                scores[n2] = scores.get(n2, 0) + s2

    results = scores.items()
    # Sort in descending order by score
    results.sort(lambda (n1,s1), (n2,s2): cmp(s2,s1))
    # n - 1 matches, each with iters * numrounds rounds.
    norm = float(iters * numrounds * (len(names)-1))
    if html:
        print "<hr><h1>Results</h1>"
        print "<h3>Average scores per round:</h3>"
        print "<table>"
        print "\n".join("<tr><td>%s</td> <td>%.2f</td><td>%s</td></tr>" % (redact(name),score / norm, agents[name].getStrategy()) for (name,score) in results)
        print "</table>"
    else:
        print "Average scores per round:"
        print "\n".join("%s: %.2f" % (
            redact(name),score / norm) for (name,score) in results)
    ifhtml("</center>")
    
            

if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception, e:
        print "ERROR: ", e
        print ""
        print sys.exc_info()
