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
    cmd_kill = "rsh %s /usr/local/sbin/%s" % (node, proc) 
    sp.Popen(cmd_kill.split(), stdout=sp.PIPE, stderr=sp.PIPE)

