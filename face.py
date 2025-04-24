from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
from datetime import datetime
import json
import base64
import uuid
import logging
from tensorflow.keras.models import load_model
import mysql.connector
from mysql.connector import Error
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Increase maximum content length to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MODELS_FOLDER'] = 'models'
app.config['FALLBACK_MODE'] = False
# Database configuration
app.config['DB_CONFIG'] = {
    'host': 'localhost',
    'user': 'root',  # Change to your MySQL username
    'password': '',  # Change to your MySQL password
    'database': 'workforce'
}
# Add security headers configurations
app.config['SECURITY_HEADERS'] = {
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; media-src 'self' blob:",
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'SAMEORIGIN',
    'Feature-Policy': "camera 'self'; microphone 'none'"
}

# Create required directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER'], app.config['MODELS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Database connection helper function
def get_db_connection():
    """Create a connection to the MySQL database"""
    try:
        conn = mysql.connector.connect(**app.config['DB_CONFIG'])
        if conn.is_connected():
            return conn
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
    return None

# Initialize database tables if they don't exist
def init_database():
    """Initialize database tables if they don't exist"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database for initialization")
        return False

    cursor = conn.cursor()
    try:
        # Create employee_dash table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS employee_dash (
            employee_id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            department VARCHAR(50) NOT NULL,
            role VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create attendance_dash table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_dash (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            check_in TIME,
            check_out TIME,
            overtime_hours FLOAT DEFAULT 0,
            status VARCHAR(20) DEFAULT 'Present',
            FOREIGN KEY (employee_id) REFERENCES employee_dash(employee_id)
        )
        ''')

        # Create face_embeddings table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(20) NOT NULL,
            embedding LONGTEXT NOT NULL,
            face_path VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employee_dash(employee_id)
        )
        ''')
        conn.commit()
        logger.info("Database tables initialized successfully")
        return True
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# Set global variable for tensorflow availability
tensorflow_available = False
facenet_model = None

# Try to import TensorFlow and load model
try:
    import tensorflow as tf
    # Set memory growth to avoid memory allocation issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    tensorflow_available = True
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    logger.error(f"TensorFlow import error: {e}")
    logger.warning("Face recognition will be limited without TensorFlow")

def load_facenet_model():
    """Load the FaceNet model if available"""
    global facenet_model, tensorflow_available

    if not tensorflow_available:
        logger.warning("Cannot load FaceNet model - TensorFlow not available")
        return False

    model_path = os.path.join(app.config['MODELS_FOLDER'], 'facenet_keras.h5')

    if os.path.exists(model_path):
        try:
            # Load model with proper error handling
            facenet_model = tf.keras.models.load_model(model_path, compile=False)
            # Warm up the model with a dummy prediction
            dummy_input = np.zeros((1, 160, 160, 3))
            facenet_model.predict(dummy_input, verbose=0)
            logger.info("FaceNet model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading FaceNet model: {e}")
            facenet_model = None
    else:
        logger.error(f"FaceNet model not found at {model_path}")
        logger.info(f"Please download facenet_keras.h5 and place it in {os.path.abspath(app.config['MODELS_FOLDER'])}")

    return False

# Try to load the model
model_loaded = load_facenet_model()

# Helper functions
def detect_face(image_path):
    """Detect face in image using OpenCV with improved parameters for better registration"""
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None, None

    # Read image in color
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None, None

    try:
        # Verify image isn't corrupt or empty
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            logger.error(f"Image is empty or corrupt: {image_path}")
            return None, None

        # Convert to grayscale for face detection only
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Load the face cascade with error checking
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            logger.error(f"Cascade classifier file not found: {face_cascade_path}")
            # Try alternative path
            face_cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(face_cascade_path):
                logger.error("Cannot find face cascade classifier file")
                return None, None

        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            logger.error("Failed to load cascade classifier")
            return None, None

        # Very lenient parameters for registration
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2,
                                             minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) == 0:
            logger.warning("No faces detected in image with default parameters")
            # Try with extremely lenient parameters
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1,
                                                minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) == 0:
                # Last try with LBP cascade which might be better for some images
                alt_cascade_path = cv2.data.haarcascades + 'lbpcascade_frontalface.xml'
                if os.path.exists(alt_cascade_path):
                    alt_cascade = cv2.CascadeClassifier(alt_cascade_path)
                    faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1,
                                                      minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

                if len(faces) == 0:
                    logger.error("No faces detected in image with any parameters")
                    # Save debug image for analysis
                    debug_path = os.path.join(app.config['TEMP_FOLDER'], f'failed_detection_{uuid.uuid4().hex[:8]}.jpg')
                    cv2.imwrite(debug_path, image)
                    logger.info(f"Saved failed detection image to {debug_path}")
                    return None, None
                else:
                    logger.info("Face detected with LBP cascade")
            else:
                logger.info("Face detected with extremely lenient parameters")
        else:
            logger.info("Face detected with default parameters")

        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Add some margin around the face (30% on each side for better recognition)
        margin_x = int(w * 0.3)
        margin_y = int(h * 0.3)

        # Make sure the margins don't go outside the image bounds
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(image.shape[1], x + w + margin_x)
        y_end = min(image.shape[0], y + h + margin_y)

        # Extract face ROI with margins from original color image
        face_roi = image[y_start:y_end, x_start:x_end]

        # For debugging, save the detected face
        debug_path = os.path.join(app.config['TEMP_FOLDER'], f'debug_face_{uuid.uuid4().hex[:8]}.jpg')
        cv2.imwrite(debug_path, face_roi)
        logger.info(f"Saved debug face image to {debug_path}")

        return face_roi, largest_face
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        # Save the image for debugging
        try:
            debug_path = os.path.join(app.config['TEMP_FOLDER'], f'error_image_{uuid.uuid4().hex[:8]}.jpg')
            cv2.imwrite(debug_path, image)
            logger.info(f"Saved error image to {debug_path}")
        except:
            pass
        return None, None

def preprocess_face(face_img):
    """Preprocess face for storage or display with improved error handling"""
    try:
        if face_img is None or face_img.size == 0:
            logger.error("Empty face image provided to preprocess_face")
            return None

        # Verify image dimensions
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            logger.error(f"Invalid face image dimensions: {face_img.shape}")
            return None

        # Resize to a standard size for FaceNet (160x160)
        face_img = cv2.resize(face_img, (160, 160))

        # Additional preprocessing: enhance contrast and brightness
        # Convert to Lab color space
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        # Split channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # Merge channels
        limg = cv2.merge((cl, a, b))
        # Convert back to BGR
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Save a debug copy of the preprocessed face
        debug_path = os.path.join(app.config['TEMP_FOLDER'], f'debug_processed_{uuid.uuid4().hex[:8]}.jpg')
        cv2.imwrite(debug_path, enhanced_img)
        logger.info(f"Saved debug processed face to {debug_path}")

        return enhanced_img
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None

def get_face_embedding(image_path):
    """Extract face embedding from image with improved error handling"""
    global facenet_model, tensorflow_available

    # Check if we can use FaceNet
    if not tensorflow_available or facenet_model is None:
        logger.warning("FaceNet not available, cannot generate real embeddings")
        return None

    # Detect face in image
    face_img, face_rect = detect_face(image_path)
    if face_img is None:
        logger.warning(f"No face detected in {image_path}")
        return None

    try:
        # Preprocess for FaceNet
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype(np.float32)
        # Normalize pixel values to range [-1, 1]
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        # Get embedding
        embedding = facenet_model.predict(face_img, verbose=0)
        return embedding[0]
    except Exception as e:
        logger.error(f"Error getting face embedding: {e}")
        return None

def get_face_embedding_from_array(face_img):
    """Extract face embedding directly from face image array"""
    global facenet_model, tensorflow_available

    # Check if we can use FaceNet
    if not tensorflow_available or facenet_model is None:
        logger.warning("FaceNet not available, cannot generate real embeddings")
        return None

    if face_img is None:
        logger.warning("No face image provided")
        return None

    try:
        # Preprocess for FaceNet
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype(np.float32)
        # Normalize pixel values to range [-1, 1]
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        # Get embedding
        embedding = facenet_model.predict(face_img, verbose=0)
        return embedding[0]
    except Exception as e:
        logger.error(f"Error getting face embedding from array: {e}")
        return None

def save_face_image(employee_id, face_img, count):
    """Save face image for an employee"""
    if face_img is None or face_img.size == 0:
        logger.error("Empty face image provided to save_face_image")
        return None

    # Ensure employee_id is valid
    if not employee_id:
        logger.error("Invalid employee ID provided to save_face_image")
        return None

    # Ensure the count is valid
    try:
        count_str = str(count)
    except Exception:
        logger.error(f"Invalid count provided: {count}")
        count_str = uuid.uuid4().hex[:8]  # Use random ID if count is invalid

    employee_dir = os.path.join(app.config['UPLOAD_FOLDER'], employee_id)
    os.makedirs(employee_dir, exist_ok=True)

    face_path = os.path.join(employee_dir, f'face_{count_str}.jpg')
    try:
        cv2.imwrite(face_path, face_img)
        logger.info(f"Attempted to save face image to {face_path}")

        # Check if the file was saved successfully
        if not os.path.exists(face_path):
            logger.error(f"File not created: {face_path}")
            return None

        if os.path.getsize(face_path) == 0:
            logger.error(f"File is empty: {face_path}")
            os.remove(face_path)  # Remove empty file
            return None

        logger.info(f"Successfully saved face image to {face_path}")
        return face_path
    except Exception as e:
        logger.error(f"Error saving face image: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    try:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return -1

def save_employee_embedding(employee_id, embedding, face_path):
    """Save employee face embedding to database"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False

    cursor = conn.cursor()
    try:
        # Convert numpy array to JSON string
        embedding_json = json.dumps(embedding.tolist())

        # Insert embedding into database
        query = """
        INSERT INTO face_embeddings (employee_id, embedding, face_path)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (employee_id, embedding_json, face_path))
        conn.commit()
        logger.info(f"Saved face embedding for employee {employee_id}")
        return True
    except Error as e:
        logger.error(f"Error saving face embedding: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_employee_embeddings(employee_id=None):
    """Get employee face embeddings from database"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return {}

    cursor = conn.cursor(dictionary=True)
    try:
        if employee_id:
            # Get embeddings for specific employee
            query = """
            SELECT e.employee_id, e.name, e.department as worker_type, f.embedding, f.face_path
            FROM face_embeddings f
            JOIN employee_dash e ON f.employee_id = e.employee_id
            WHERE f.employee_id = %s
            """
            cursor.execute(query, (employee_id,))
        else:
            # Get all embeddings
            query = """
            SELECT e.employee_id, e.name, e.department as worker_type, f.embedding, f.face_path
            FROM face_embeddings f
            JOIN employee_dash e ON f.employee_id = e.employee_id
            """
            cursor.execute(query)

        results = cursor.fetchall()

        # Process results
        embeddings_data = {}
        for row in results:
            employee_id = row['employee_id']
            if employee_id not in embeddings_data:
                embeddings_data[employee_id] = {
                    'name': row['name'],
                    'worker_type': row['worker_type'],
                    'embeddings': [],
                    'face_paths': []
                }

            # Convert JSON string back to numpy array
            embedding = np.array(json.loads(row['embedding']))
            embeddings_data[employee_id]['embeddings'].append(embedding)
            embeddings_data[employee_id]['face_paths'].append(row['face_path'])

        return embeddings_data
    except Error as e:
        logger.error(f"Error getting face embeddings: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

def recognize_face(face_embedding, threshold=0.5):
    """Recognize face from database using cosine similarity"""
    if face_embedding is None:
        logger.warning("Cannot recognize: no face embedding provided")
        return None

    # Get all employee embeddings from database
    embeddings_data = get_employee_embeddings()

    if not embeddings_data:
        logger.warning("No employees with face data in database")
        return None

    best_match = None
    best_similarity = -1

    # Compare with all stored embeddings
    for employee_id, employee_data in embeddings_data.items():
        # Compare with all stored embeddings for this employee
        for embedding in employee_data['embeddings']:
            # Calculate cosine similarity
            similarity = calculate_similarity(face_embedding, embedding)
            logger.info(f"Similarity with {employee_id} ({employee_data['name']}): {similarity}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (employee_id, employee_data['name'], employee_data['worker_type'])

    # Return best match if similarity is above threshold
    if best_similarity > threshold:
        logger.info(f"Face recognized as {best_match[1]} with similarity {best_similarity}")
        return best_match
    else:
        logger.warning(f"Face not recognized. Best similarity was {best_similarity} with threshold {threshold}")
        return None

def record_attendance(employee_id, name):
    """Record attendance in database"""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False

    cursor = conn.cursor()
    try:
        # Check if attendance already recorded today
        query = """
        SELECT id FROM attendance_dash
        WHERE employee_id = %s AND date = %s
        """
        cursor.execute(query, (employee_id, date_str))
        existing = cursor.fetchone()

        if existing:
            logger.info(f"Attendance already recorded today for {name}")
            return False

        # Insert new attendance record
        query = """
        INSERT INTO attendance_dash (employee_id, date, check_in, check_out, overtime_hours, status)
        VALUES (%s, %s, %s, NULL, 0, 'Present')
        """
        cursor.execute(query, (employee_id, date_str, time_str))
        conn.commit()
        logger.info(f"Attendance recorded for {name}")
        return True
    except Error as e:
        logger.error(f"Error recording attendance: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def update_check_out(employee_id):
    """Update check-out time and calculate overtime hours"""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False, "Database connection failed"

    cursor = conn.cursor(dictionary=True)
    try:
        # Check if attendance record exists for today
        query = """
        SELECT id, check_in FROM attendance_dash
        WHERE employee_id = %s AND date = %s
        """
        cursor.execute(query, (employee_id, date_str))
        record = cursor.fetchone()

        if not record:
            logger.warning(f"No check-in record found for employee {employee_id} on {date_str}")
            return False, "No check-in record found for today"

        # Calculate overtime hours (assuming standard 8-hour workday)
        check_in_time = datetime.strptime(record['check_in'].strftime('%H:%M:%S'), '%H:%M:%S')
        check_out_time = datetime.strptime(time_str, '%H:%M:%S')

        # Calculate hours worked (in hours)
        time_diff = check_out_time - check_in_time
        hours_worked = time_diff.total_seconds() / 3600

        # Calculate overtime (hours worked beyond 8 hours)
        overtime_hours = max(0, hours_worked - 8)
        overtime_hours = round(overtime_hours, 1)  # Round to one decimal place

        # Update attendance record
        query = """
        UPDATE attendance_dash
        SET check_out = %s, overtime_hours = %s
        WHERE id = %s
        """
        cursor.execute(query, (time_str, overtime_hours, record['id']))
        conn.commit()

        logger.info(f"Check-out recorded for employee {employee_id} with {overtime_hours} overtime hours")
        return True, f"Check-out recorded with {overtime_hours} overtime hours"
    except Error as e:
        logger.error(f"Error updating check-out: {e}")
        return False, f"Error updating check-out: {str(e)}"
    finally:
        cursor.close()
        conn.close()

def get_employee_by_id(employee_id):
    """Get employee details from database"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return None

    cursor = conn.cursor(dictionary=True)
    try:
        query = """
        SELECT * FROM employee_dash WHERE employee_id = %s
        """
        cursor.execute(query, (employee_id,))
        employee = cursor.fetchone()
        return employee
    except Error as e:
        logger.error(f"Error getting employee: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_all_employees():
    """Get all employees from database"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        query = """
        SELECT employee_id, name, department as worker_type, role
        FROM employee_dash
        ORDER BY name
        """
        cursor.execute(query)
        employees = cursor.fetchall()

        # Count face embeddings for each employee
        for employee in employees:
            try:
                sub_query = """
                SELECT COUNT(*) as face_count
                FROM face_embeddings
                WHERE employee_id = %s
                """
                cursor.execute(sub_query, (employee['employee_id'],))
                result = cursor.fetchone()
                employee['faces_registered'] = result['face_count'] if result else 0
            except:
                employee['faces_registered'] = 0

        return employees
    except Error as e:
        logger.error(f"Error getting employees: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def get_attendance_records(from_date=None, to_date=None, employee_id=None):
    """Get attendance records from database"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        query_params = []
        query = """
        SELECT a.*, e.name, e.department as worker_type
        FROM attendance_dash a
        JOIN employee_dash e ON a.employee_id = e.employee_id
        WHERE 1=1
        """

        if from_date:
            query += " AND a.date >= %s"
            query_params.append(from_date)

        if to_date:
            query += " AND a.date <= %s"
            query_params.append(to_date)

        if employee_id:
            query += " AND a.employee_id = %s"
            query_params.append(employee_id)

        query += " ORDER BY a.date DESC, a.check_in DESC"

        cursor.execute(query, query_params)
        records = cursor.fetchall()

        # Format datetime objects to strings
        for record in records:
            if 'check_in' in record and record['check_in']:
                record['check_in'] = record['check_in'].strftime('%H:%M:%S')
            if 'check_out' in record and record['check_out']:
                record['check_out'] = record['check_out'].strftime('%H:%M:%S')
            if 'date' in record and record['date']:
                record['date'] = record['date'].strftime('%Y-%m-%d')

        return records
    except Error as e:
        logger.error(f"Error getting attendance records: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# Security header middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to responses"""
    for header, value in app.config['SECURITY_HEADERS'].items():
        response.headers[header] = value
    return response

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/system_status')
def system_status():
    """Get system status including mode and model availability"""
    worker_count = 0
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM employee_dash")
            worker_count = cursor.fetchone()[0]
        except:
            pass
        finally:
            cursor.close()
            conn.close()

    return jsonify({
        'success': True,
        'fallback_mode': app.config['FALLBACK_MODE'],
        'tensorflow_available': tensorflow_available,
        'model_loaded': facenet_model is not None,
        'database_connected': get_db_connection() is not None,
        'worker_count': worker_count
    })

@app.route('/get_workers')
def get_workers():
    """Get all employees"""
    employees = get_all_employees()
    return jsonify({'success': True, 'workers': employees})

@app.route('/delete_worker', methods=['POST'])
def delete_worker():
    try:
        data = request.get_json()
        worker_id = data.get('worker_id')

        if not worker_id:
            return jsonify({'success': False, 'message': 'Missing worker ID'})

        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})

        cursor = conn.cursor()
        try:
            # First delete face embeddings
            query = "DELETE FROM face_embeddings WHERE employee_id = %s"
            cursor.execute(query, (worker_id,))

            # Delete attendance records
            query = "DELETE FROM attendance_dash WHERE employee_id = %s"
            cursor.execute(query, (worker_id,))

            # Finally delete employee record
            query = "DELETE FROM employee_dash WHERE employee_id = %s"
            cursor.execute(query, (worker_id,))

            conn.commit()

            # Delete associated face image files
            employee_dir = os.path.join(app.config['UPLOAD_FOLDER'], worker_id)
            if os.path.exists(employee_dir):
                import shutil
                shutil.rmtree(employee_dir)

            return jsonify({'success': True, 'message': f'Worker {worker_id} deleted successfully'})
        except Error as e:
            conn.rollback()
            logger.error(f"Error deleting worker: {e}")
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'})
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Exception in delete_worker: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/register_worker', methods=['POST'])
def register_worker():
    """
    Register a new worker with face images.
    """
    try:
        # Get data from request body
        data = request.get_json()
        if not data:
            logger.error("No data provided in register_worker request")
            return jsonify({'success': False, 'message': 'No data provided'})

        # Extract worker information
        worker_id = data.get('worker_id')
        name = data.get('name')
        worker_type = data.get('worker_type')
        face_images = data.get('face_images', [])

        # Validate required fields
        if not worker_id or not name or not worker_type:
            logger.error(f"Missing required worker information: id={worker_id}, name={name}, type={worker_type}")
            return jsonify({'success': False, 'message': 'Missing required worker information'})

        if len(face_images) == 0:
            logger.error("No face images provided for worker registration")
            return jsonify({'success': False, 'message': 'No face images provided'})

        logger.info(f"Registering worker: {worker_id}, {name}, {worker_type} with {len(face_images)} images")

        # Insert or update worker in database
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed during worker registration")
            return jsonify({'success': False, 'message': 'Database connection failed'})

        cursor = conn.cursor()
        try:
            # Check if worker already exists
            query = "SELECT employee_id FROM employee_dash WHERE employee_id = %s"
            cursor.execute(query, (worker_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing worker
                logger.info(f"Updating existing worker: {worker_id}")
                query = """
                UPDATE employee_dash
                SET name = %s, department = %s, role = %s
                WHERE employee_id = %s
                """
                cursor.execute(query, (name, worker_type, worker_type, worker_id))
            else:
                # Insert new worker
                logger.info(f"Inserting new worker: {worker_id}")
                query = """
                INSERT INTO employee_dash (employee_id, name, department, role)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (worker_id, name, worker_type, worker_type))

            conn.commit()
        except Error as e:
            logger.error(f"Error registering worker in database: {e}")
            return jsonify({'success': False, 'message': f'Error registering worker: {str(e)}'})
        finally:
            cursor.close()
            conn.close()

        # Process face images
        success_count = 0
        for i, image_data in enumerate(face_images):
            try:
                logger.info(f"Processing face image {i+1}/{len(face_images)} for worker {worker_id}")

                # Skip the data URL prefix
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                # Decode base64 to image
                image_binary = base64.b64decode(image_data)
                temp_path = os.path.join(app.config['TEMP_FOLDER'], f'temp_face_{worker_id}_{i}.jpg')

                with open(temp_path, 'wb') as f:
                    f.write(image_binary)

                logger.debug(f"Saved face image to {temp_path}")

                # Detect face in image
                face_img, face_rect = detect_face(temp_path)

                if face_img is not None:
                    logger.info(f"Face detected in image {i+1}")

                    # Process and save face
                    processed_face = preprocess_face(face_img)
                    if processed_face is not None:
                        # Save face image
                        face_path = save_face_image(worker_id, processed_face, i)
                        if face_path is not None:
                            # Get face embedding
                            embedding = get_face_embedding_from_array(processed_face)
                            if embedding is not None:
                                # Save embedding to database
                                if save_employee_embedding(worker_id, embedding, face_path):
                                    success_count += 1
                                    logger.info(f"Successfully processed and saved face {i+1}")
                                else:
                                    logger.error(f"Failed to save embedding for face {i+1}")
                            else:
                                logger.error(f"Failed to generate embedding for face {i+1}")
                        else:
                            logger.error(f"Failed to save face image {i+1}")
                    else:
                        logger.error(f"Failed to preprocess face {i+1}")
                else:
                    logger.warning(f"No face detected in image {i+1}")

                # Clean up temp file
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_path}: {e}")

            except Exception as e:
                logger.error(f"Error processing face image {i+1}: {e}")

        if success_count == 0:
            logger.error(f"Failed to process any faces for worker {worker_id}")
            return jsonify({
                'success': False,
                'message': 'Failed to process any face images. Please try again with clearer images.'
            })

        logger.info(f"Worker {worker_id} registered successfully with {success_count} face images")
        return jsonify({
            'success': True,
            'message': f'Worker registered with {success_count} face images'
        })

    except Exception as e:
        logger.exception("Worker registration error")
        return jsonify({'success': False, 'message': f'Error during registration: {str(e)}'})

@app.route('/check_worker_id', methods=['GET'])
def check_worker_id():
    """
    Check if a worker ID already exists in the database.
    """
    try:
        worker_id = request.args.get('worker_id')
        if not worker_id:
            return jsonify({'success': False, 'message': 'Worker ID is required'})

        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})

        cursor = conn.cursor()
        try:
            query = "SELECT COUNT(*) FROM employee_dash WHERE employee_id = %s"
            cursor.execute(query, (worker_id,))
            count = cursor.fetchone()[0]

            return jsonify({
                'success': True,
                'exists': count > 0
            })
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.exception("Error checking worker ID")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
@app.route('/check_face', methods=['POST'])
def check_face():
    """
    Check if a face is present in the provided image and assess quality.
    """
    try:
        # Get image from request
        if 'image' not in request.files:
            # Try to get base64 image
            image_data = request.form.get('image')
            if not image_data or not image_data.startswith('data:image'):
                logger.error("No image data provided to check_face")
                return jsonify({'success': False, 'message': 'No image provided'})

            # Convert base64 to image file
            try:
                # Extract the actual base64 data after the comma
                image_data = image_data.split(',')[1]
                image_binary = base64.b64decode(image_data)

                temp_path = os.path.join(app.config['TEMP_FOLDER'], f'temp_check_{uuid.uuid4().hex}.jpg')
                with open(temp_path, 'wb') as f:
                    f.write(image_binary)
                logger.debug(f"Saved base64 image to {temp_path}")
            except Exception as e:
                logger.error(f"Error decoding base64 image: {e}")
                return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})
        else:
            image_file = request.files['image']
            temp_path = os.path.join(app.config['TEMP_FOLDER'], f'temp_check_{uuid.uuid4().hex}.jpg')
            image_file.save(temp_path)
            logger.debug(f"Saved uploaded image to {temp_path}")

        # Check if face is present in image using OpenCV
        face_img, face_rect = detect_face(temp_path)

        # Clean up temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

        # Assess face quality if requested
        quality_score = 0
        assess_quality = request.form.get('assess_quality') == 'true'

        if face_img is not None and assess_quality:
            # Simple quality assessment based on image size and clarity
            quality_score = min(1.0, (face_img.shape[0] * face_img.shape[1]) / (160*160))
            # Adjust based on brightness and contrast
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            # Normalize to 0-1 range for ideal values
            brightness_score = 1.0 - abs((brightness - 128) / 128)
            contrast_score = min(1.0, contrast / 50)
            quality_score = 0.4 * quality_score + 0.3 * brightness_score + 0.3 * contrast_score
            logger.debug(f"Face quality assessment: {quality_score:.2f}")

        if face_img is not None:
            logger.info("Face detected successfully")
            return jsonify({
                'success': True,
                'face_detected': True,
                'quality_score': quality_score if assess_quality else None,
                'quality': quality_score if assess_quality else None  # Added for compatibility
            })
        else:
            logger.warning("No face detected in image")
            return jsonify({
                'success': True,
                'face_detected': False,
                'message': 'No face detected. Please try again with better lighting and make sure your face is clearly visible.'
            })
    except Exception as e:
        logger.exception("Face check error")
        return jsonify({'success': False, 'message': f'Error processing request: {str(e)}'})

@app.route('/recognize_worker', methods=['POST'])
def recognize_worker():
    """
    Recognize a worker from a face image and record attendance.
    """
    try:
        # Get image from the request
        if 'image' not in request.files:
            # Try to get base64 image
            image_data = request.form.get('image')
            if not image_data or not image_data.startswith('data:image'):
                return jsonify({'success': False, 'message': 'No image provided'})

            # Convert base64 to image file
            try:
                # Extract the actual base64 data after the comma
                image_data = image_data.split(',')[1]
                image_binary = base64.b64decode(image_data)

                temp_path = os.path.join(app.config['TEMP_FOLDER'], f'temp_recognize_{uuid.uuid4().hex}.jpg')
                with open(temp_path, 'wb') as f:
                    f.write(image_binary)
            except Exception as e:
                logger.error(f"Error decoding base64 image: {e}")
                return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})
        else:
            image_file = request.files['image']
            temp_path = os.path.join(app.config['TEMP_FOLDER'], f'temp_recognize_{uuid.uuid4().hex}.jpg')
            image_file.save(temp_path)

        # Check if we have TensorFlow available
        if not tensorflow_available or facenet_model is None:
            try:
                os.remove(temp_path)
            except:
                pass
            logger.warning("Cannot recognize faces - TensorFlow/FaceNet not available")
            return jsonify({'success': False, 'message': 'Face recognition not available in this mode'})

        # Get embedding from image
        embedding = get_face_embedding(temp_path)

        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass

        if embedding is None:
            logger.warning("Failed to get face embedding for recognition")
            return jsonify({'success': False, 'message': 'No face detected or failed to process face'})

        # Recognize face
        result = recognize_face(embedding, threshold=0.5)
        if result is None:
            logger.warning("Face not recognized")
            return jsonify({
                'success': True,
                'worker_found': False,
                'message': 'Worker not recognized'
            })

        worker_id, name, worker_type = result
        logger.info(f"Face recognized as {name} ({worker_id})")

        # Record attendance
        now = datetime.now()
        attendance_recorded = record_attendance(worker_id, name)

        return jsonify({
            'success': True,
            'worker_found': True,
            'worker_id': worker_id,
            'name': name,
            'worker_type': worker_type,
            'status': 'Checked In' if attendance_recorded else 'Already Checked In',
            'time': now.strftime('%H:%M:%S'),
            'message': f'Recognition successful: {name}'
        })

    except Exception as e:
        logger.exception("Recognition error")
        return jsonify({'success': False, 'message': f'Error during recognition: {str(e)}'})

@app.route('/add_face', methods=['POST'])
def add_face():
    """Add additional face images for an existing worker"""
    try:
        worker_id = request.form.get('worker_id')
        if not worker_id:
            return jsonify({'success': False, 'message': 'Worker ID required'})

        # Verify worker exists
        worker = get_employee_by_id(worker_id)
        if not worker:
            return jsonify({'success': False, 'message': f'Worker ID {worker_id} not found'})

        # Process face image
        face_image = request.files.get('face_image')
        if not face_image:
            return jsonify({'success': False, 'message': 'No face image provided'})

        # Save uploaded image temporarily
        temp_filename = os.path.join(app.config['TEMP_FOLDER'], f"{uuid.uuid4().hex}.jpg")
        face_image.save(temp_filename)

        # Detect face in image
        face_img, face_rect = detect_face(temp_filename)
        if face_img is None:
            os.remove(temp_filename)
            return jsonify({'success': False, 'message': 'No face detected in image'})

        # Process detected face
        processed_face = preprocess_face(face_img)
        if processed_face is None:
            os.remove(temp_filename)
            return jsonify({'success': False, 'message': 'Error processing face image'})

        # Get current face count
        conn = get_db_connection()
        face_count = 0
        if conn:
            cursor = conn.cursor()
            try:
                query = "SELECT COUNT(*) FROM face_embeddings WHERE employee_id = %s"
                cursor.execute(query, (worker_id,))
                face_count = cursor.fetchone()[0]
            finally:
                cursor.close()
                conn.close()

        # Save processed face
        face_path = save_face_image(worker_id, processed_face, face_count + 1)
        if not face_path:
            os.remove(temp_filename)
            return jsonify({'success': False, 'message': 'Error saving face image'})

        # Generate face embedding if model is available
        success = False
        if tensorflow_available and facenet_model is not None:
            embedding = get_face_embedding_from_array(processed_face)
            if embedding is not None:
                # Save embedding to database
                success = save_employee_embedding(worker_id, embedding, face_path)

        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass

        if success:
            return jsonify({
                'success': True,
                'message': f'Face added for {worker["name"]}',
                'face_path': face_path,
                'face_count': face_count + 1
            })
        else:
            return jsonify({'success': False, 'message': 'Error saving face embedding'})

    except Exception as e:
        logger.error(f"Exception in add_face: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/recognize', methods=['POST'])
def recognize():
    """Recognize face in uploaded image"""
    try:
        # Check if face recognition is available
        if not tensorflow_available or facenet_model is None:
            return jsonify({
                'success': False,
                'message': 'Face recognition not available in fallback mode'
            })

        # Get image from request
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image provided'})

        image = request.files['image']
        if image.filename == '':
            return jsonify({'success': False, 'message': 'Empty file'})

        # Save uploaded image temporarily
        temp_filename = os.path.join(app.config['TEMP_FOLDER'], f"{uuid.uuid4().hex}.jpg")
        image.save(temp_filename)

        # Get face embedding
        face_embedding = get_face_embedding(temp_filename)

        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass

        if face_embedding is None:
            return jsonify({'success': False, 'message': 'No face detected in image'})

        # Recognize face
        recognition_result = recognize_face(face_embedding)
        if recognition_result:
            employee_id, name, worker_type = recognition_result

            # Determine if check-in or check-out
            mode = request.form.get('mode', 'check-in')

            if mode == 'check-out':
                success, message = update_check_out(employee_id)
                if success:
                    return jsonify({
                        'success': True,
                        'message': f'Check-out successful: {name}',
                        'name': name,
                        'employee_id': employee_id,
                        'worker_type': worker_type,
                        'check_out_message': message
                    })
                else:
                    return jsonify({
                        'success': False,
                        'recognized': True,
                        'message': message,
                        'name': name
                    })
            else:  # check-in
                attendance_result = record_attendance(employee_id, name)
                return jsonify({
                    'success': True,
                    'message': f'Face recognized: {name}',
                    'name': name,
                    'employee_id': employee_id,
                    'worker_type': worker_type,
                    'attendance_recorded': attendance_result
                })
        else:
            return jsonify({
                'success': False,
                'message': 'Face not recognized'
            })
    except Exception as e:
        logger.error(f"Exception in recognize: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    """Get attendance records with optional filters"""
    try:
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')
        employee_id = request.args.get('employee_id')

        records = get_attendance_records(from_date, to_date, employee_id)
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        logger.error(f"Exception in get_attendance: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files with proper security checks"""
    # Validate filename to prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return "Access denied", 403

    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

# Main entry point
if __name__ == '__main__':
    # Initialize database
    init_database()

    # Determine if we're in fallback mode
    app.config['FALLBACK_MODE'] = not model_loaded

    # Run the app
    app.run(debug=True, port="5003")