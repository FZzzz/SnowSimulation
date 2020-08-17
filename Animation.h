#ifndef _ANIMATION_H_
#define _ANIMATION_H_

#include "Joint.h"
#include "common.h"
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * [Caution]
 * - Channels stores rotation as degree
 *
**/

struct Channel
{
	enum class Channel_Type
	{
		X_TRANSLATION,
		Y_TRANSLATION,
		Z_TRANSLATION,
		X_ROTATION,
		Y_ROTATION,
		Z_ROTATION
	}type;
	float value;
};

struct FrameData
{
	bool movable;
	glm::vec3 translation = glm::vec3(0);
	glm::quat quaternion = glm::quat();
};

struct Frame
{
	uint32_t frame_id;
	std::unordered_map<std::shared_ptr<Joint>, std::vector<Channel>> joint_channel_map;
	std::unordered_map<std::shared_ptr<Joint>, FrameData> joint_framedata_map;
};

// getCurrentFrameData() specific frame -> step() to next frame
// getCurrentFrameTime(); --> for synchronization
// 

class Animation
{
public:
	Animation();
	~Animation();

	void Pause();
	void Step();

	void setFrames(std::vector<Frame>& frames);
	void setCurrentFrame(size_t frame_count);

	inline const std::vector<Frame>& getFrames() { return m_frames; };
	inline const Frame& getCurrentFrame() { return m_frames[m_current_frame]; };
	inline const size_t getCurrentFrameTime() { return m_current_frame; };
	inline const size_t getKeyFrameSize() { return m_frames.size(); };

	float m_frame_time;
	uint32_t m_frame_count;

private:

	// Sub-function of setFrames()
	void PrecomputeFrameData();

	std::vector<Frame> m_frames;
	size_t m_current_frame;
	bool m_pause;
	
};

#endif