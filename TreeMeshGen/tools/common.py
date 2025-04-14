import math


class vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return f"vec3({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)

    def __abs__(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return vec3(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)

    def normalized(self):
        return self / abs(self)

    def angle(self, other):
        return math.acos(self.dot(other) / (abs(self) * abs(other)))

    def rotate(self, axis, angle):
        axis = axis.normalized()
        u = axis * self.dot(axis)
        w = self - u
        v = axis.cross(w)
        return u + w * math.cos(angle) + v * math.sin(angle)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    

def process_config(config):
    """
    Check and update the config parameters for Height and DBH.
    
    - If both Height and DBH are provided, use them directly.
    - If Height is missing but DBH is provided, compute Height using H = K * DBH^Y.
    """
    if "Height" not in config and "DBH" in config:
        if "K" not in config or "Y" not in config:
            raise ValueError("Missing parameters K and Y for height computation.")
        config["Height"] = config["K"] * (config["DBH"] ** config["Y"])
    elif "Height" in config and "DBH" not in config:
        if "K" not in config or "Y" not in config:
            raise ValueError("Missing parameters K and Y for DBH computation.")
        config["DBH"] = (config["Height"] / config["K"]) ** (1 / config["Y"])
        #print (f"DBH computed from Height: {config['DBH']}")
    return config