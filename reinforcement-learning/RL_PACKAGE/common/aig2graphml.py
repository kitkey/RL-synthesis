import networkx as nx
#import dgl
import argparse,os,re
INPUT_BENCH = None
GML_DUMP_LOC = None

#############################################################
#Node types: 0-PI, 1-PO, 2-Internal
#Gate type: 0-Unassigned, 1-NOT, 2-AND, 0-BUFF
# 0- Node type, 1- Gate type, 2- Predecessors, 3- Successors
#############################################################

nodeType = {
    "PI" : 0,
    "PO" : 1,
    "Internal":2
}

edgeType = {
    "BUFF" : 0,
    "NOT" : 1,
}

def setGlobalAndEnvironmentVars(benchFile, gmlDumpLoc):
    if not (os.path.exists(benchFile)):
        print("Paths are invalid. Please rerun")
        exit(1)

    global INPUT_BENCH, GML_DUMP_LOC
    INPUT_BENCH = benchFile
    GML_DUMP_LOC = gmlDumpLoc


def processANDAssignments(inputs,output,idxCounter,poList,nodeNameIDMapping,singleGateInputIOMapping,AIG_DAG):
    nType = nodeType["Internal"]
    nodeAttributedDict = {
        "node_id": output,
        "node_type": nType,
        "num_inverted_predecessors": 0
    }
    AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
    nodeNameIDMapping[output] = idxCounter
    numInvertedPredecessors = 0
    for inp in inputs:
        if not (inp in nodeNameIDMapping.keys()):
            srcIdx = nodeNameIDMapping[singleGateInputIOMapping[inp]]
            eType = edgeType["NOT"]
            numInvertedPredecessors+=1
        else:
            srcIdx = nodeNameIDMapping[inp]
            eType = edgeType["BUFF"]
        AIG_DAG.add_edge(idxCounter,srcIdx,edge_type=eType)
    AIG_DAG.nodes[idxCounter]["num_inverted_predecessors"] = numInvertedPredecessors

    # If output is primary output, add additional node to keep it consistent with POs having inverters
    if (output in poList):
        nType = nodeType["PO"]
        nodeAttributedDict = {
            "node_id": output+"_buff",
            "node_type": nType,
            "num_inverted_predecessors": 0
        }
        AIG_DAG.add_nodes_from([(idxCounter+1, nodeAttributedDict)])
        nodeNameIDMapping[output+"_buff"] = idxCounter+1
        srcIdx = idxCounter
        eType = edgeType["BUFF"]
        AIG_DAG.add_edge(idxCounter+1,srcIdx, edge_type=eType)


def parseAIGBenchAndCreateNetworkXGraph():
    nodeNameIDMapping = {}
    singleInputgateIOMapping = {}
    poList = []
    benchFile = open(INPUT_BENCH,'r+')
    benchFileLines = benchFile.readlines()
    #print(benchFileLines[0:100])
    benchFile.close()
    AIG_DAG = nx.DiGraph()
    idxCounter = 0
    for line in benchFileLines:
        if len(line) == 0 or line.__contains__("ABC"):
            continue
        elif line.__contains__("vdd"):
            ## Treat Vdd assignment as Primary Input.
            line = line.replace(" ","")
            pi = re.search("(.*?)=", str(line)).group(1)
            nodeAttributedDict = {
                "node_id": pi,
                "node_type": nodeType["PI"],
                "num_inverted_predecessors": 0
            }
            AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
            nodeNameIDMapping[pi] = idxCounter
            idxCounter+=1
        elif line.__contains__("INPUT"):
            line = line.replace(" ","")
            pi = re.search("INPUT\((.*?)\)",str(line)).group(1)
            nodeAttributedDict = {
                "node_id": pi,
                "node_type": nodeType["PI"],
                "num_inverted_predecessors": 0
            }
            AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
            nodeNameIDMapping[pi] = idxCounter
            idxCounter+=1
        elif line.__contains__("OUTPUT"):
            line = line.replace(" ", "")
            po = re.search("OUTPUT\((.*?)\)", str(line)).group(1)
            poList.append(po)
        elif line.__contains__("AND"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            input1 = re.search("AND\((.*?),",str(line)).group(1)
            input2 = re.search(",(.*?)\)", str(line)).group(1)
            processANDAssignments([input1,input2], output, idxCounter, poList, nodeNameIDMapping, singleInputgateIOMapping, AIG_DAG)
            if output in poList:
                idxCounter += 1
            idxCounter+=1
        elif line.__contains__("NOT"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            inputPin = re.search("NOT\((.*?)\)", str(line)).group(1)
            singleInputgateIOMapping[output] = inputPin
            if output in poList:
                nodeAttributedDict = {
                    "node_id": output+"_inv",
                    "node_type": nodeType["PO"],
                    "num_inverted_predecessors": 1
                }
                AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
                nodeNameIDMapping[output+"_inv"] = idxCounter
                srcIdx = nodeNameIDMapping[inputPin]
                eType = edgeType["NOT"]
                AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
                idxCounter += 1
        elif line.__contains__("BUFF"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            inputPin = re.search("BUFF\((.*?)\)", str(line)).group(1)
            singleInputgateIOMapping[output] = inputPin
            numInvertedPredecessors = 0
            if output in poList:
                ## Additional logic: Input pin may be inverter. So perform an initial check whether inputPin in available
                ## in nodeNameIDMapping or not
                if inputPin in nodeNameIDMapping.keys():
                    srcIdx = nodeNameIDMapping[inputPin]
                    eType = edgeType["BUFF"]
                else:
                    ## instance of NOT gate followed by BUFF gates.
                    srcIdx = nodeNameIDMapping[singleInputgateIOMapping[inputPin]]
                    eType = edgeType["NOT"]
                    numInvertedPredecessors+=1
                nodeAttributedDict = {
                    "node_id": output+"_buff",
                    "node_type": nodeType["PO"],
                    "num_inverted_predecessors": numInvertedPredecessors
                }
                AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
                nodeNameIDMapping[output+"_buff"] = idxCounter
                AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
                idxCounter += 1
        else:
            print(" Line contains unknown characters.",line)
            exit(1)
    return AIG_DAG


def dumpGMLGraph(nxCktDAG):
    gmlfileName = os.path.basename(INPUT_BENCH)+".graphml"
    nx.write_graphml(nxCktDAG, os.path.join(GML_DUMP_LOC,gmlfileName))


def parseCmdLineArgs(bench=None, gml=None):
    parser = argparse.ArgumentParser(prog='AIGBENCH2GML', description="AIG bench to GML converter")
    parser.add_argument('--version', action='version', version='1.0.0')
    
    # Изменяем аргументы на необязательные
    parser.add_argument('--bench', required=bench is None, default=bench, help="Path of AIG bench File")
    parser.add_argument('--gml', required=gml is None, default=gml, help="GML file dump location")
    
    # Если аргументы bench и gml переданы, используем их, иначе парсим из командной строки
    if bench is not None and gml is not None:
        return argparse.Namespace(bench=bench, gml=gml)
    else:
        return parser.parse_args()
