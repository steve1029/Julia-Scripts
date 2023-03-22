import subprocess as sp
import time

nodes = []
homename = 'yhome%01d'
nodename = 'y%03d'

dhcpd = open('/etc/dhcp/dhcpd.conf', 'r').read()
dhcpd_split = dhcpd.split()
#print dhcpd_split

mac_address = [] 
hostnames   = []
while True:
	try:
		idx = dhcpd_split.index('ethernet')
		mac = dhcpd_split[idx+1].replace(';', '')
		hostname = dhcpd_split[idx-3]
		mac_address.append(mac)
		hostnames.append(hostname)
		dhcpd_split[idx] = ''
	except:
		break

def ext_exec_wait(cmd):
	out, err = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
	if err != '': print err
	return out

def ext_exec_nowait(cmd):
	sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)

for i, mac in enumerate(mac_address):
	print 'etherwake %s' % hostnames[i]
	command = '/usr/sbin/etherwake %s' % mac
        ext_exec_wait(command)
	time.sleep(5)

