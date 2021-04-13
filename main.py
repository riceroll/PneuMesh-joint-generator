import sys
import os

import numpy as np
import json
import polyscope
from tqdm import tqdm
from scipy.optimize import minimize


numPointsOnCurve = 10       # number of points on a curve without center
kSpring = 0.8
kAngle = 20
kCenter = 0
kRepulsion = 0.1

channelRadius = 0.3
l0 = 0.1
h = 0.1

show = False
if len(sys.argv) >= 2:
    show = bool(int(sys.argv[1]))

if len(sys.argv) >= 3:
    kRepulsion = float(sys.argv[2])

if len(sys.argv) >= 4:
    kCenter = float(sys.argv[3])

isForced = False
if len(sys.argv) >= 5:
    isForced = bool(sys.argv[4])
    
numSteps = 1000
if len(sys.argv) >= 6:
    numSteps = int(sys.argv[5])
    

def norm(vecs):
    vecs = np.array(vecs)
    vecs = vecs.reshape(-1, 3)
    norms = np.sqrt(np.sum(vecs ** 2, 1))
    return norms

class Vertex:
    all = []
    
    def __init__(self, pos):
        # pos: position, array-like, [3, ]
        self.pos = np.array(pos)
        self.edges = set()
        Vertex.all.append(self)
    
    def neighbor_vertices(self):
        vs = []
        cs = []     # channels
        for e in self.edges:
            vs.append(e.other_vertex(self))
            cs.append(e.iChannel)
        return vs, cs
    
    def out_unit_vectors(self):
        # unit vectors pointing from self to neighbor vertices
        vs, cs = self.neighbor_vertices()
        vecs = []
        for v in vs:
            vec = v.pos - self.pos
            vecs.append(vec / norm(vec))
        return vecs, cs


class Edge:
    all = []
    
    def __init__(self, iv0, iv1, iChannel):
        v0 = Vertex.all[iv0]
        v1 = Vertex.all[iv1]
        self.vs = {v0, v1}
        self.iChannel = iChannel
        
        v0.edges.add(self)
        v1.edges.add(self)
        Edge.all.append(self)
    
    def other_vertex(self, v):
        return list(self.vs - {v})[0]
    

def createChannels(v):
    # collect channels for a vertex
    # input: Vertex
    # output:
    #   a list of channels: [ channel:[point, ...], ...]
    #   centers corresponding to each channel:  [ point, ...]
    
    vecs, cs = v.out_unit_vectors()
    vecs = np.array(vecs)

    # collect all channels of a vertex
    channels = []
    centers = []
    for iChannel in sorted(list(set(cs))):
        cs = np.array(cs)
        ids = np.where(cs == iChannel)[0]
    
        channel = []
        center = (np.random.random(3) - 0.5) * 0.01
    
        points = vecs[ids]  # np.array: n x 3
        for point in points:
            ts = np.arange(numPointsOnCurve) / numPointsOnCurve
            ts = ts.reshape(-1, 1)
            point = point.reshape(1, 3)
            # points on the channel without the center: n x 3, the last closest to the center
            curve = (1 - ts) * point + ts * center
            channel.append(curve)
    
        channels.append(channel)
        centers.append(center)

    return channels, centers

def addConstraints(channels, centers):
    # convert channels and centers to points and constraints
    # input:
    #   channels : a list of channels: [ channel:[point, ...], ...]
    #   centers : #   centers corresponding to each channel:  [ point, ...]
    # output:
    #   points:  all the points on channels, np.array n x 3
    #   edgeConstraints: pairs of point indices connected by edge constraints (on the same channel)
    #   angleConstraints:  triplet of point indices, points are consecutive on the same curve
    #   repulsionConstraints: pairs of point ids on different channels

    points = []  # nv x 3
    edgeConstraints = []  # ne x 2
    iChannel = [] # ne x 1
    angleConstraints = []  # na x 3
    repulsionConstraints = []  # nr x 2
    fixConstraints = []     # id of points at the ends, nf
    fixedPoints = []        # position of points at the ends, nf x 3
    iCenters = []       # ids of centers  nc

    iPointsChannel = []  # ids of points in each channel: [[id, ...], ...]

    for i, channel in enumerate(channels):
        iPoints = []
    
        center = centers[i]
        iCenter = len(points)
        iCenters.append(iCenter)
        points.append(center)
    
        iPoints.append(iCenter)
    
        for curve in channel:
            for i, point in enumerate(curve):
            
                iPoint = len(points)
                points.append(point)
                iPoints.append(iPoint)
            
                if i == 0:
                    fixConstraints.append(iPoint)
                    fixedPoints.append(point)
            
                if i != len(curve) - 1:
                    iPointNext = len(points)
                
                    edgeConstraints.append([iPoint, iPointNext])
                else:
                    edgeConstraints.append([iPoint, iCenter])
                iChannel.append(iCenter)
    
        iPointsChannel.append(iPoints)

    points = np.array(points)
    edgeConstraints = np.array(edgeConstraints)
    fixConstraints = np.array(fixConstraints)
    fixedPoints = np.array(fixedPoints)
    iCenters = np.array(iCenters)

    for i in range(len(channels)):
        for j in np.arange(i + 1, len(channels)):
            if i == j:
                continue
            for iPoint in iPointsChannel[i]:
                for jPoint in iPointsChannel[j]:
                    repulsionConstraints.append([iPoint, jPoint])
    repulsionConstraints = np.array(repulsionConstraints)
    
    ecNext = None
    for i, ec in enumerate(edgeConstraints):
        if i != len(edgeConstraints) - 1:
            ecNext = edgeConstraints[i+1]
            if ec[1] == ecNext[0]:
                angleConstraints.append([ec[0], ec[1], ecNext[1]])
    angleConstraints = np.array(angleConstraints)

    return points, edgeConstraints, iChannel, angleConstraints, repulsionConstraints, fixConstraints, fixedPoints, iCenters

# def step(ps, fs):
#     # update location of ps for each step

def step(ps, es, ans, rs, fs, fps, ics):
    # optimize points locations with repulsion and edge constraints
    # input:
    #   ps: all the points on channels, np.array n x 3
    #   es: pairs of point indices connected by edge constraints (on the same channel), np.array ne x 2
    #   ans: triplet of point indices, points are consecutive on the same curve. np.array na x 3
    #   rs: pairs of point ids on different channels, np.array nr x 2
    #   fs: ids of points with fixed position, np.array nf
    #   fps: pos of points with fixed position, np.array nf x 3
    #   ics: ids of center points
    # output:
    #   ps: points with updated positions, np.array n x 3

    ps = ps.reshape(-1, 3)
    v = np.zeros_like(ps)
    f = np.zeros_like(ps)

    vec_e01 = ps[es[:, 1]] - ps[es[:, 0]]
    
    ed = norm(vec_e01).reshape(-1, 1)  # edgeDistances, ne, 1
    fe0 = kSpring * ed * vec_e01 / ed
    fe1 = -fe0
    
    fc = kCenter * -ps[ics]
    
    ff = fps - ps[fs]
    
    np.add.at(f, es[:, 0], fe0)
    np.add.at(f, es[:, 1], fe1)
    
    if len(rs) > 0:
        vec_r01 = ps[rs[:, 1]] - ps[rs[:, 0]]
        rd = norm(vec_r01).reshape(-1, 1)  # repulsionDistances, nr, 1

        if (rd != 0).all():
            # idsVeryClose = rs[np.where(rd < channelRadius * 1.5)[0]]
            # idsClose = rs[np.where(rd < channelRadius * 2)[0]]
            fr0vc = kRepulsion * -vec_r01 / rd * channelRadius * 0.1  # nr x 3
            fr1vc = kRepulsion * -fr0vc
            fr0c = kRepulsion * -vec_r01 / rd * channelRadius * 0.02
            fr1c = kRepulsion * -fr0vc
        
            fr0vc = fr0vc * (rd < channelRadius * 1.5)
            fr1vc = fr1vc * (rd < channelRadius * 1.5)
            fr0c = fr0c * (rd < channelRadius * 2)
            fr1c = fr1c * (rd < channelRadius * 2)
            np.add.at(f, rs[:, 0], fr0vc)
            np.add.at(f, rs[:, 1], fr1vc)
            np.add.at(f, rs[:, 0], fr0c)
            np.add.at(f, rs[:, 1], fr1c)
        else:
            print('rd == 0')
    
    # angle constraints
    if False:
        ip0 = ans[:, 1]
        ip1 = ans[:, 0]
        ip2 = ans[:, 2]
        p0 = ps[ip0]
        p1 = ps[ip1]
        p2 = ps[ip2]
        vec01 = p1 - p0
        vec02 = p2 - p0
        n = np.cross(vec02, vec01)
        
        cos = np.sum(vec01 * vec02, 1) / np.sqrt(np.sum(vec01 ** 2, 1)) / np.sqrt(np.sum(vec02 ** 2, 1))
        cos[ cos >= 1] = 1 - 1e-6
        cos[ cos <= - 1] = -1 + 1e-6
        alpha = np.arccos(cos).reshape(-1, 1)
        
        deltaAlpha = alpha - np.pi
        fa1 = kAngle * deltaAlpha * np.cross(n, vec01)
        fa2 = kAngle * deltaAlpha * np.cross(n, vec02)
        fa0 = kAngle * deltaAlpha * (np.cross(n, vec02) - np.cross(n, vec01))
        
        fa1[fa1 > 1] = 1
        fa1[fa1 < -1] = -1
        fa2[fa2 > 1] = 1
        fa2[fa2 < -1] = -1
        fa0[fa0 > 1] = 1
        fa0[fa0 < -1] = -1
        
        np.add.at(f, ip0, fa0)
        np.add.at(f, ip1, fa1)
        np.add.at(f, ip2, fa2)
    
    # force for center points (drag them back)
    # np.add.at(f, ics, fc)
    np.add.at(f, ics, fc)
    
    # force for fixed points (drag them back)
    # np.add.at(f, fs, ff)
    f[fs, :] *= 0
    
    v += f * h
    ps += v * h
    
    # print(np.sum(norm(f)))
    
    # energy = 0
    # energy += 1 * np.sum(ed ** 2)
    # energy += 5000 * np.sum((rd < channelRadius * 2) * (channelRadius * 2 - rd) ** 2)
    # energy += 100 * np.sum(norm(ps[fs] - fps))
    
    # energy /= 1 / 1000 * (len(ed) + len(rd))
    
    # return energy
    
    
    
def optimize(ps, es, rs, fs, fps):
    energy = lambda x: step(x, es, rs, fs, fps)
    
    res = minimize(energy, ps, method='nelder-mead',
             options={
                 'maxiter': 3e4,
                 'fatol': 10,
                 'disp': True,
                 'adaptive': True
             })
    return res.x.reshape(-1, 3)

def vertexToChannelCurves(v):
    # usage:
    #   Turn a joint vertex to channels.
    #   Each channel has multiple curves and at least one center intersection point
    #   Each curve, represented by a list of nodes, connect the center to a point on the unit sphere
    # input: Vertex
    # output: a list of curves, [ curve:[point, ...], ...]
    
    channels, centers = createChannels(v)
    points, edgeConstraints, iChannel, angleConstraints, repulsionConstraints, fixConstraints, fixedPoints, iCenters = addConstraints(channels, centers)
    
    return points, edgeConstraints, iChannel, angleConstraints, repulsionConstraints, fixConstraints, fixedPoints, iCenters
    
directory = os.path.dirname(os.path.realpath(__file__)) + '/jsons/'
    
for fileName in os.listdir(directory):
    Vertex.all = []
    Edge.all = []

    if fileName.split('.')[-1] != 'json':
        continue
    print('Processing '+fileName)
    fileName = directory + fileName

    # main
    with open(fileName) as ifile:
        content = ifile.read()
        data = json.loads(content)

    if 'channel' in data and not isForced:
        print('Already processed.')
        continue
    vs = data['v']
    es = data['e']
    try:
        esChannel = data['edgeChannel']
    except:
        esChannel = [0] * len(es)

    for pos in vs:
        Vertex(pos)

    for i, e in enumerate(es):
        Edge(e[0], e[1], esChannel[i])

    idsChannel = list(set(esChannel))
    idsChannel.sort()

    if show:
        polyscope.init()

    data['channel'] = []

    for v in tqdm(Vertex.all):
        ps, es, ic, ans, rs, fs, fps, ics = vertexToChannelCurves(v)
        ps_cloud = polyscope.register_point_cloud("my points", ps)
        ps_net = polyscope.register_curve_network("my network", ps, es)

        all_ic = list(set(ic))
        ic = [all_ic.index(i) for i in ic]

        cs = [np.array([0.7,0.4,0.2]) if i==0 else np.array([0.4, 0.7, 0.2]) if i==1 else np.array([0.4,0.2,0.7]) for i in ic]
        cs = np.array(cs)
        # print(cs)

        # data['channel'].append({'ps': ps.tolist(), 'es': es.tolist()})

        # ps_cloud.update_point_positions(ps)
        # ps_net.update_node_positions(ps)


        # ps_net.add_color_quantity("val", cs, defined_on='edges')
        # if show:
        #     polyscope.show()

        for j in range(1):
          for i in range(numSteps):
              step(ps, es, ans, rs, fs, fps, ics)

              
          data['channel'].append({'ps': ps.tolist(), 'es': es.tolist()})

          ps_cloud.update_point_positions(ps)
          ps_net.update_node_positions(ps)


          ps_net.add_color_quantity("val", cs, defined_on='edges')
          if show:
              polyscope.show()

    with open(fileName, 'w') as ofile:
        content = json.dumps(data)
        ofile.write(content)

