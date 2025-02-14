import pymssql

class Car:
    license_plate: str
    is_parked_legally: bool
    parking_place: int
    prev_pos: tuple[int, int]
    
    def __init__(self, p, l, pp, prev):
        self.is_parked_legally = None
        self.license_plate = p
        self.parking_place = pp
        self.prev_pos = prev

def write_number_to_db(number):
    server = '.'
    user = 'parking1'
    password = 'parking'
    conn = pymssql.connect(server, user, password, "parking")
    cursor = conn.cursor()

    cursor.execute('''DECLARE @retval BIT;
    EXEC INSERT_PLATE @NUMBER = %s, @RESULT = @retval OUTPUT;

    SELECT @retval''', number)
    rv = False
    if (cursor.fetchone()[0] == 1):
        rv = True
    conn.commit()

    conn.close()
    return rv

def park(car: Car):
    server = '.'
    user = 'parking1'
    password = 'parking'
    conn = pymssql.connect(server, user, password, "parking")
    cursor = conn.cursor()

    cursor.execute('''EXEC PARK @NUMBER = %s, @ID = %s, @LEGALLY=%s''', (car.license_plate, car.parking_place, car.is_parked_legally))
    conn.commit()

    conn.close()

def unpark(car: Car):
    server = '.'
    user = 'parking1'
    password = 'parking'
    conn = pymssql.connect(server, user, password, "parking")
    cursor = conn.cursor()

    cursor.execute('''EXEC UNPARK @NUMBER = %s''', car.license_plate)
    conn.commit()

    conn.close()

def exit_parking(car: Car):
    server = '.'
    user = 'parking1'
    password = 'parking'
    conn = pymssql.connect(server, user, password, "parking")
    cursor = conn.cursor()

    cursor.execute('''EXEC EXITING @NUMBER = %s''', car.license_plate)
    conn.commit()

    conn.close()