use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

fn euclidean_distance(a: &Point, b: &Point) -> f64 {
    ((a.x - b.x).powf(2.0) + (a.y - b.y).powf(2.0)).sqrt()
}

fn find_nearest_cluster(point: &Point, centroids: &[Point]) -> usize {
    let mut min_distance = std::f64::INFINITY;
    let mut nearest_cluster = 0;

    for (i, centroid) in centroids.iter().enumerate() {
        let distance = euclidean_distance(point, centroid);
        if distance < min_distance {
            min_distance = distance;
            nearest_cluster = i;
        }
    }

    nearest_cluster
}

fn compute_centroids(points: &[Point], clusters: &[usize], k: usize) -> Vec<Point> {
    let mut centroids = vec![Point { x: 0.0, y: 0.0 }; k];
    let mut counts = vec![0; k];

    for (point, &cluster) in points.iter().zip(clusters.iter()) {
        centroids[cluster].x += point.x;
        centroids[cluster].y += point.y;
        counts[cluster] += 1;
    }

    for i in 0..k {
        if counts[i] > 0 {
            centroids[i].x /= counts[i] as f64;
            centroids[i].y /= counts[i] as f64;
        } else {
            centroids[i] = Point {
                x: rand::thread_rng().gen_range(0.0, 1.0),
                y: rand::thread_rng().gen_range(0.0, 1.0),
            };
        }
    }

    centroids
}

fn kmeans(points: &[Point], k: usize, max_iterations: usize) -> (Vec<usize>, Vec<Point>) {
    let mut centroids = vec![Point { x: 0.0, y: 0.0 }; k];
    let mut clusters = vec![0; points.len()];

    for i in 0..k {
        centroids[i] = points[rand::thread_rng().gen_range(0, points.len())].clone();
    }

    for _ in 0..max_iterations {
        for (i, point) in points.iter().enumerate() {
            clusters[i] = find_nearest_cluster(point, &centroids);
        }

        let new_centroids = compute_centroids(points, &clusters, k);

        if centroids == new_centroids {
            return (clusters, centroids);
        }

        centroids = new_centroids;
    }

    (clusters, centroids)
}

fn main() {
    let points = vec![
        Point { x: 0.1, y: 0.1 },
        Point { x: 0.15, y: 0.2 },
        Point { x: 0.2, y: 0.15 },
        Point { x: 0.8, y: 0.8 },
        Point { x: 0.7, y: 0.9 },
        Point { x: 0.9, y: 0.7 },
    ];

    let k = 2;
    let max_iterations = 100;

    let (clusters, centroids) = kmeans(&points, k, max_iterations);

    println!("Clusters: {:?}", clusters);
    println!("Centroids: {:?}", centroids);
}
