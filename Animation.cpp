#include "Animation.h"

Animation::Animation() : m_pause(false), m_current_frame(0), m_frame_count(0)
{
}


Animation::~Animation()
{
}

void Animation::Pause()
{
	m_pause = !m_pause;
}

void Animation::Step()
{
	if (m_pause)
		return;
	m_current_frame = (m_current_frame < m_frame_count - 1) ? (m_current_frame + 1): 0;
}

void Animation::setFrames(std::vector<Frame>& frames)
{
	m_frames = frames;
	PrecomputeFrameData();
}

void Animation::setCurrentFrame(size_t frame_count)
{
	m_current_frame = frame_count;
}

void Animation::PrecomputeFrameData()
{
	if (m_frames.size() == 0)
		return;
	for (size_t i = 0; i < m_frames.size(); ++i)
	{

		std::unordered_map<std::shared_ptr<Joint>, FrameData> joint_framedata_map;

		for (auto map_it = m_frames[i].joint_channel_map.begin();
			map_it != m_frames[i].joint_channel_map.end();
			map_it++)
		{
			bool movable = false;
			auto joint = map_it->first;
			auto channels = map_it->second;
			const float deg_to_rad = M_PI / 180.0f;

			FrameData frame_data;

			glm::vec3 translation = glm::vec3(0, 0, 0);
			glm::mat4 rot_mat(1);

			// Resolve channel
			for (size_t j = 0; j < channels.size(); ++j)
			{
				switch (channels[j].type)
				{
				case Channel::Channel_Type::X_TRANSLATION:
				{
					translation.x = channels[j].value;
					movable = true;
					break;
				}
				case Channel::Channel_Type::Y_TRANSLATION:
				{
					translation.y = channels[j].value;
					movable = true;
					break;
				}
				case Channel::Channel_Type::Z_TRANSLATION:
				{
					translation.z = channels[j].value;
					movable = true;
					break;
				}
				case Channel::Channel_Type::X_ROTATION:
				{
					rot_mat = glm::rotate(rot_mat, channels[j].value * deg_to_rad, glm::vec3(1, 0, 0));// *rot_mat;
					break;
				}
				case Channel::Channel_Type::Y_ROTATION:
				{
					rot_mat = glm::rotate(rot_mat, channels[j].value * deg_to_rad, glm::vec3(0, 1, 0));// *rot_mat;
					break;
				}
				case Channel::Channel_Type::Z_ROTATION:
				{
					rot_mat = glm::rotate(rot_mat, channels[j].value * deg_to_rad, glm::vec3(0, 0, 1));// *rot_mat;
					break;
				}

				}

			}// end of channel for loop

			frame_data.movable = movable;
			if (movable)
				frame_data.translation = translation;

			glm::quat rot_quat = glm::quat_cast(rot_mat);
			frame_data.quaternion = rot_quat;

			joint_framedata_map.emplace(joint, frame_data);

		} // end of map iteration
		
		m_frames[i].joint_framedata_map.clear();
		m_frames[i].joint_framedata_map = joint_framedata_map;

	} // end of frames loop
	
}
