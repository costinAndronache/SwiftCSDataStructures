//  Write some awesome Swift code, or import libraries like "Foundation",
//  "Dispatch", or "Glibc"

public struct Queue<T> {
    
    private var queue: [T] = []
    
    public mutating func enqueue(_ val: T){
        self.queue.append(val)
    }
    
    public mutating func dequeue() -> T? {
        guard let val: T = self.queue.first else {
            return nil
        }
        self.queue.remove(at: 0)
        return val
    }
    
	public mutating func peek() -> T? {
		return self.queue.first
	}
	
    var isEmpty: Bool {
        return self.queue.count == 0
    }
}

public struct Stack<T> {
	private var stack: [T] = []
	
	public mutating func push(_ val: T) {
		self.stack.append(val)
	}
	
	public mutating func pop() -> T? {
		if let last = self.stack.last {
			self.stack.removeLast()
			return last
		}
		
		return nil
	}
	
	public func peek() -> T? {
		return self.stack.last
	}
}

public struct Heap<T: Comparable, U: Hashable> {
	
	public enum HeapType {
		case maxHeap
		case minHeap
	}
	
	public struct Item<T, U> {
		var priority: T
		let payload: U
	}
	
	private typealias CompareFunction<T> = (T, T) -> Bool 
	private let firstShouldBeParentOfSecond: CompareFunction<T>
	private var heap: [Item<T, U>] = []
	private var indexForPayload: [U: Int] = [:]
	
	public init(type: HeapType){
		if type == .maxHeap {
			self.firstShouldBeParentOfSecond = { $0 >= $1 }
		} else {
			self.firstShouldBeParentOfSecond = { $0 <= $1 }
		}
	}
	
	public func priorityFor(payload: U) -> T? {
		guard let index = self.indexForPayload[payload] else {
			return nil
		}
		
		return self.heap[index].priority
	}
	
	public mutating func insert(item: Item<T,U>){
		self.heap.append(item)
		self.indexForPayload[item.payload] = self.heap.count - 1
		self.percolate(self.heap.count - 1)
	}
	
	public mutating func peekTop() -> Item<T,U>? {
		return heap.first
	}
	
	public mutating func removeTop() {
		guard self.heap.count > 0 else {
			return
		}
		
		self.heap[0] = self.heap[self.heap.count - 1]
		self.heap.removeLast()
		self.sift(0)
	}
	
	private mutating func sift(_ index: Int){
		var parent = index
		while(true) {
			guard let bestSon = self.bestSon(parent) else {
				return
			}
			
			if self.firstShouldBeParentOfSecond(self.heap[bestSon].priority, 
										 self.heap[parent].priority) {
				
				self.swapNodes(parent, bestSon)
				parent = bestSon
			} else {
				return
			}
		}
	}
	
	public mutating func changePriority(of payload: U, to priority: T){
		guard let index = self.indexForPayload[payload] else {
			print("could not find index for \(payload)")
			return
		}
		
		
		self.heap[index].priority = priority
		self.percolate(index)
		self.sift(index)
	}
	
	private mutating func swapNodes(_ i: Int, _ j: Int){
		self.indexForPayload[self.heap[i].payload] = j
		self.indexForPayload[self.heap[j].payload] = i
		swap(&self.heap[i], &self.heap[j])
	}
	
	private mutating func percolate(_ index: Int){
		let priority = self.heap[index].priority
		var node = index
		
		while (node > 0 && (self.firstShouldBeParentOfSecond(priority,
							self.heap[self.parent(node)].priority))){
			
			
			self.swapNodes(node, self.parent(node))
			node = self.parent(node)
		}
	}
	
	private func parent(_ index: Int) -> Int { return (index-1) / 2}
	private func leftChild(_ index: Int) -> Int? {
		let child = 2 * index + 1
		if child < self.heap.count {
			return child
		}
		return nil
	}
	private func rightChild(_ index: Int) -> Int? {
		guard let leftChild = self.leftChild(index), 
			 leftChild + 1 < self.heap.count else {
			return nil
		}
		
		return leftChild + 1
	}
	
	
	
	private func bestSon(_ parent: Int) -> Int? {
		if let left = self.leftChild(parent),
		 let right = self.rightChild(parent){
			 if firstShouldBeParentOfSecond(self.heap[left].priority,
									  self.heap[right].priority){
					   return left;
				   }
				   return right;
				}
			
				if let left = self.leftChild(parent) {return left}
				if let right = self.rightChild(parent) {return right}
				return nil
	}
	
	public func printDebug() {
		for i in stride(from: 0, to: self.heap.count, by: 1) {
			print("[\(i)] = (payload: \(self.heap[i].payload), priority:\(self.heap[i].priority))")
		}
	}
}



public struct DisjointSetsForest<T: Hashable> {
	private var parent: [T: T] = [:]
	private var level: [T: Int] = [:]
	
	
	public mutating func makeSetFor(_ value: T) {
		parent[value] = value
		level[value] = 1
	}
	
	public mutating func inSameSet(_ x: T, _ y: T) -> Bool {
		guard let rx = representativeFor(x),
			  let ry = representativeFor(y),
			  rx == ry else {
				  return false
			  }
		return true
	}
	
	public mutating func uniteInSameSet(_ x: T, _ y: T){
		guard let x = representativeFor(x), 
			  let y = representativeFor(y) else {
			return
		}
		
		let xLevel = levelFor(x)
		let yLevel = levelFor(y)
		
		if xLevel > yLevel {
			parent[y] = x 
		} else {
			parent[x] = y
		}
		
		if xLevel == yLevel {
			level[y] = yLevel + 1
		}
	}
	
	private mutating func levelFor(_ value: T) -> Int {
		return self.level[value] ?? 0
	}
	
	private mutating func representativeFor(_ value: T) -> T? {
		var currentValue = value
		while let ancestor = parent[currentValue], 
				  ancestor != currentValue {
			currentValue = ancestor
		}
		
		
			var x = value
			while let ancestor = parent[x], 
				  ancestor != currentValue {
				  parent[x] = currentValue
				  x = ancestor
			}
		
		
		return currentValue
	}
}

public struct Edge: Hashable {
	public let from: Int
	public let to: Int
	
	public var hashValue: Int {
		return "\(from)\(to)".hashValue
	} 
}

public func ==(lhs: Edge, rhs: Edge) -> Bool {
	return lhs.from == rhs.from && 
		   lhs.to == rhs.to
}

public struct WeightedEdge: Hashable {
	public let edge: Edge
	public let cost: Double
	
	public var hashValue: Int {
		return edge.hashValue
	}
}

public func ==(lhs: WeightedEdge, rhs: WeightedEdge) -> Bool {
	return lhs.edge == rhs.edge
}

public protocol GraphProtocol {
	func neighboursFor(node: Int) -> [Int]
	func degreeFor(node: Int) -> Int?
	func allNodes() -> [Int]
	func allEdges() -> [Edge]
}

public protocol WeightedGraphProtocol: GraphProtocol {
	func costForEdge(from: Int, to: Int) -> Double?
	func allEdgesWeighted() -> [WeightedEdge]
}


public struct WeightedGraph: WeightedGraphProtocol {
    
    private var adjancencyListPerNode: [Int: [Int]] = [:]
    private var costMapPerNode: [Int: [Int: Double]] = [:]
	private let type: GraphType
	
	public init(type: GraphType){
		self.type = type
	}
	
	public enum GraphType {
		case directed
		case undirected
	}
	
	public func degreeFor(node: Int) -> Int? {
		return self.adjancencyListPerNode[node]?.count
	}
	
	public func allNodes() -> [Int] {
		return Array<Int>(self.adjancencyListPerNode.keys)
	}
	
	public func allEdges() -> [Edge] {
		var edgesSet: Set<Edge> = Set<Edge>()
		
		self.allNodes().forEach { node in 
			let neighbours = self.neighboursFor(node: node)
			neighbours.forEach {
				let edge = Edge(from: node, to: $0)
				if self.type == .directed {
					edgesSet.insert(edge)
				} else if !edgesSet.contains(Edge(from: $0, to: node)) {
					edgesSet.insert(edge)
				}
			}
		}
		
		return Array<Edge>(edgesSet)
	}
	
	public func allEdgesWeighted() -> [WeightedEdge]{
		return self.allEdges().map({
			return WeightedEdge(edge: $0, cost: self.costForEdge($0)!)
		})
	}
	
    public mutating func addNodeIfNotExisting(_ node: Int){
        if let _ = self.adjancencyListPerNode[node] {
            return
        }
        
        self.adjancencyListPerNode[node] = []
		self.costMapPerNode[node] = [:]
    }
    
    public mutating func addEdge(from: Int, to: Int, cost: Double = 0.0){
        self.addNodeIfNotExisting(from)
        self.addNodeIfNotExisting(to)
        
        if !(self.adjancencyListPerNode[from]?.contains(to) ?? true){
                    self.adjancencyListPerNode[from]?.append(to)
			self.changeCost(from: from, to: to, newCost: cost)
        }
        
		
        if self.type == .undirected &&
		!(self.adjancencyListPerNode[to]?.contains(from) ?? true){
                    self.adjancencyListPerNode[to]?.append(from)
			self.changeCost(from: to, to: from, newCost: cost)
        }
    }
    
	public mutating func changeCost(from: Int, to: Int, newCost: Double){
		self.costMapPerNode[from]?[to] = newCost
	}
    
	public func neighboursFor(node: Int) -> [Int] {
		return self.adjancencyListPerNode[node] ?? []
	}
	
	public func costForEdge(from: Int, to: Int) -> Double? {
		return self.costMapPerNode[from]?[to]
	}
	
    public func costForEdge(_ edge: Edge) -> Double? {
		return self.costForEdge(from: edge.from, to: edge.to)
	}

}

public struct DFS {
	public static func dfs(_ parentsPerNode: [Int: [Int]], _ node: Int, _ pathsArray: inout [[Int]], _ currentPath: inout [Int]){
        let parents = parentsPerNode[node] ?? []
        
        currentPath.append(node)
        if parents.count == 0 {
            pathsArray.append(currentPath.reversed())
        } 
        
        parents.forEach {
            dfs(parentsPerNode, $0, &pathsArray, &currentPath)
        }
        
        currentPath.remove(at: currentPath.count - 1)
    }
}

public struct BFS {
	
	public static func allShortestPathsIn(graph: GraphProtocol, from: Int, to: Int) -> [[Int]] {
		
		var result: [[Int]] = []
        
        var distancePerNode: [Int: Int] = [:]
        var parentsForNode: [Int: [Int]] = [:]
        
        let distanceForNode: (Int) -> Int  = {
            return distancePerNode[$0] ?? -1
        }
        
        graph.allNodes().forEach {parentsForNode[$0] = []}
		
        var queue: Queue<Int> = Queue<Int>()
        queue.enqueue(from)
		distancePerNode[from] = 0
		
        var done: Bool = false 
        while(!queue.isEmpty){
            if let currentNode = queue.dequeue(){ 
               let neighbours = graph.neighboursFor(node: currentNode)
               
               neighbours.forEach {
                   if distanceForNode($0) == -1 {
                       distancePerNode[$0] = distanceForNode(currentNode) + 1
                       queue.enqueue($0)
                       parentsForNode[$0]?.append(currentNode)
                   } else if distanceForNode($0) == distanceForNode(currentNode) + 1 {
                       parentsForNode[$0]?.append(currentNode)
					   
                   }
                   
                   if $0 == to {
                       done = true
                   }
               }
            }
        }
        
        if done {
            var path: [Int] = []
            DFS.dfs(parentsForNode, to, &result, &path)
        }
        
        return result
	}
}


public struct ShortestPath {
	
	public enum Result {
			case negativeOrNoCostFound(from: Int, to: Int)
			case result(distancePerNode: [Int: Double])
		}
	
	public static func Dijkstra(graph: WeightedGraphProtocol, maxCost: Double, from: Int,  to: Int? = nil) -> Result {
		
		var distanceForNode: [Int: Double] = [:]
		var heap: Heap<Double, Int> = Heap<Double, Int>(type: .minHeap)
		let dist: (Int) -> Double = {
			return distanceForNode[$0] ?? maxCost
		}
		
		graph.allNodes().forEach {
			distanceForNode[$0] = maxCost
			heap.insert(item: Heap<Double, Int>.Item(priority: maxCost, payload: $0))
		}
		
		distanceForNode[from] = 0.0
		heap.changePriority(of: from, to: 0)
		
		while let best = heap.peekTop() {
			heap.removeTop()
			let node = best.payload
			
			let neighbours = graph.neighboursFor(node: node)
			for neighbour in neighbours {
				guard let cost = graph.costForEdge(from: node, to: neighbour), cost > 0.0 else {
					return .negativeOrNoCostFound(from: node, to: neighbour)
				}

				let bestDistance = dist(node) + cost
				if (dist(neighbour) > bestDistance){
					distanceForNode[neighbour] = bestDistance
					heap.changePriority(of: neighbour, to: bestDistance)
				}
				
				if let to = to, neighbour == to {
					return .result(distancePerNode: distanceForNode)
				}
			}
		}
		
		return .result(distancePerNode: distanceForNode)
	}
}


public struct Connectivity {
	public static func isConnected(graph: GraphProtocol) -> Bool {
		var visited: Set<Int> = Set<Int>()
		let nodes = graph.allNodes()
		guard let firstNode = nodes.first else {
			return false
		}
		
		var queue = Queue<Int>()
		queue.enqueue(firstNode)
		while let node = queue.dequeue() {
			let neighbours = graph.neighboursFor(node: node)
			for u in neighbours {
				if !visited.contains(u) {
					visited.insert(u)
					queue.enqueue(u)
				}
			}
		}
		
		for v in nodes {
			if !visited.contains(v) {
				return false
			}
		}
		
		return true
	}
}


public struct Cycles {
	public enum EulerianCycleResult {
		case graphNotConnected
		case oddDegreeForNode(node: Int)
		case result(nodesOrder: [Int])
	}
	
	public static func eulerianCycleIn(graph: GraphProtocol) -> EulerianCycleResult {
		guard Connectivity.isConnected(graph: graph) else {
			return .graphNotConnected
		}
		
		let nodes = graph.allNodes()
		let firstOddDegreeNode = nodes.first(where: {(graph.degreeFor(node: $0) ?? 0) % 2 == 1})
		guard firstOddDegreeNode == nil else {
			return .oddDegreeForNode(node: firstOddDegreeNode!)
		}
		
		
		//will continue later
		return EulerianCycleResult.result(nodesOrder: [])
	}
}


public struct Math {
	public static func gcd(_ x: Int, _ y: Int) -> Int {
		var a = max(x, y)
		var b = min(x, y)

		var r = a % b
		while(r != 0 ) {
			a = b
			b = r  
			r = a % b
		}
	
		return b
	}

	public static func lcm(_ a: Int, _ b: Int) -> Int {
		return (a*b)/(Math.gcd(a,b))
	}
	
	public struct Fraction: Hashable {
		public let numerator: Int
		public let denominator: Int 
		public init?(numerator: Int, denominator: Int){
			let d = Math.gcd(numerator, denominator)
			self.numerator = numerator / d
			self.denominator = denominator / d
		}
		
		public var hashValue: Int {
			return numerator ^ denominator
		}
	}
}

public func ==(lhs: Math.Fraction, rhs: Math.Fraction) -> Bool {
	return lhs.numerator == rhs.numerator && 
		   lhs.denominator == rhs.denominator
}
