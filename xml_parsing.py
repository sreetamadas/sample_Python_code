
# write in xml file

def generate_routefile():
    """ this module creates the .rou.xml route file / demand file with specifed vehicle demand values """
    random.seed(42)  # make tests reproducible
    N = 150 #3600  # number of time steps
    # demand per second from different directions
    #pWE = 1. / 10
    #pEW = 1. / 11
    #pNS = 1. / 30
    #pSN = 1. / 20
    # demand per second from different directions; increasing denominator reduces demand
    pWE = 1. /40 #1. / 10
    pEW = 1. /50 #1. / 11
    pNS = 1. /90 #1. / 30
    pSN = 1. /110 #1. / 20
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="right_dn" edges="51o 1i 3o 53i" />
        <route id="right_up" edges="51o 1i 4o 54i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="left_dn" edges="52o 2i 3o 53i" />
        <route id="left_up" edges="52o 2i 4o 54i" />
        <route id="up" edges="53o 3i 4o 54i" />
        <route id="up_lf" edges="53o 3i 1o 51i" />
        <route id="up_ri" edges="53o 3i 2o 52i" />
        <route id="down" edges="54o 4i 3o 53i" />
        <route id="down_lf" edges="54o 4i 2o 52i" />
        <route id="down_ri" edges="54o 4i 1o 51i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_dn_%i" type="typeWE" route="right_dn" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_up_%i" type="typeWE" route="right_up" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_dn_%i" type="typeWE" route="left_dn" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_up_%i" type="typeWE" route="left_up" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_lf_%i" type="typeNS" route="down_lf" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_ri_%i" type="typeNS" route="down_ri" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_lf_%i" type="typeNS" route="up_lf" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_ri_%i" type="typeNS" route="up_ri" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            
        print("</routes>", file=routes)
    
    
    
### read from xml
generate_routefile()

# parse route file
route_lane_map = {}
for route in sumolib.xml.parse("./data/cross.rou.xml", 'route'):
    #print(route.edges, route.id)
    if route.id == 'right':
        edge_ids = route.edges.split()
        route_lane_map = { i : 'WE' for i in edge_ids }
    elif route.id == 'left':
        edge_ids = route.edges.split()
        #route_lane_map.update( i : 'WE' for i in edge_ids )
        for i in edge_ids:
            route_lane_map[i] = 'WE'
    elif route.id == 'up':
        edge_ids = route.edges.split()
        #route_lane_map.update( i : 'NS' for i in edge_ids )
        for i in edge_ids:
            route_lane_map[i] = 'NS'
    elif route.id == 'down':
        edge_ids = route.edges.split()
        #route_lane_map.update( i : 'NS' for i in edge_ids )
        for i in edge_ids:
            route_lane_map[i] = 'NS'
print(route_lane_map)
