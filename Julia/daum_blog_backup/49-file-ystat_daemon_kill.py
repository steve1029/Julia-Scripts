import subprocess as sp

nodes = []

for i in xrange(1, 4):
    nodes.append('yhome%01d' % i)
for i in xrange(101, 166):
    nodes.append('y%03d' % i)
for i in xrange(201, 206):
    nodes.append('y%03d' % i)

proc = 'ystat_daemon'

for node in nodes:
    print node
    cmd_kill = "rsh %s ps -elf | grep '%s' | awk '{print $4}' | xargs kill" % (node, proc) 
    out, err = sp.Popen(cmd_kill.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    if err != '': print err

