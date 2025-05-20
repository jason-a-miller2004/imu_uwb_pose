import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
  a = '/home/lichard/Projects/imu_uwb_pose/data/raw/FootPoser_filtered/evan/activities1/evan_activities_1_sensor_orientation_filtered.pkl'
  with open(a, 'rb') as file:
    data = pickle.load(file)
  
  print(data.keys())
  plt.plot(data['left_altitude'])
  plt.plot(data['left_altitude_filtered'])
  plt.show()